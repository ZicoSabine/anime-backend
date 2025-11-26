from pathlib import Path
import csv
import json
from datetime import datetime
from typing import List, Optional
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, field_validator
from fastapi import Query


# ================================
#  MODEL DEFINITIONS (GMF + MLP)
# ================================

class GMF(nn.Module):
    """
    Corrected GMF (Generalized Matrix Factorization)
    """
    def __init__(self, n_users, n_items, embed_dim):
        super().__init__()
        self.user_embed = nn.Embedding(n_users, embed_dim)
        self.item_embed = nn.Embedding(n_items, embed_dim)
        nn.init.normal_(self.user_embed.weight, std=0.1)
        nn.init.normal_(self.item_embed.weight, std=0.1)

    def forward(self, user_idx, item_idx):
        u = self.user_embed(user_idx)
        i = self.item_embed(item_idx)
        return u * i     # FIXED from "u * 1"


class MLP(nn.Module):
    def __init__(self, n_users, n_items, layers=(64, 32, 16, 8)):
        super().__init__()
        embed_dim = layers[0] // 2
        self.user_embed = nn.Embedding(n_users, embed_dim)
        self.item_embed = nn.Embedding(n_items, embed_dim)
        nn.init.normal_(self.user_embed.weight, std=0.1)
        nn.init.normal_(self.item_embed.weight, std=0.1)

        modules = []
        input_size = layers[0]
        for out_size in layers[1:]:
            modules.append(nn.Linear(input_size, out_size))
            modules.append(nn.ReLU())
            modules.append(nn.Dropout(p=0.2))
            input_size = out_size

        self.mlp = nn.Sequential(*modules)

    def forward(self, user_idx, item_idx):
        u = self.user_embed(user_idx)
        i = self.item_embed(item_idx)
        x = torch.cat([u, i], dim=-1)
        return self.mlp(x)


class NeuMF(nn.Module):
    def __init__(self, n_users, n_items, gmf_dim=32, mlp_layers=(64, 32, 16, 8)):
        super().__init__()
        self.gmf = GMF(n_users, n_items, gmf_dim)
        self.mlp = MLP(n_users, n_items, layers=mlp_layers)
        self.predict_layer = nn.Linear(gmf_dim + mlp_layers[-1], 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, user_idx, item_idx):
        gmf_vec = self.gmf(user_idx, item_idx)
        mlp_vec = self.mlp(user_idx, item_idx)
        x = torch.cat([gmf_vec, mlp_vec], dim=-1)
        out = self.predict_layer(x)
        out = self.sigmoid(out).squeeze(-1)
        return out


# ================================
#  PATHS & DEVICE
# ================================

BASE_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"

CKPT_PATH = ARTIFACTS_DIR / "neumf.pt"
ITEM_META_PATH = ARTIFACTS_DIR / "item_meta.csv"
USER_MAP_PATH = ARTIFACTS_DIR / "user_map.csv"
FEEDBACK_PATH = ARTIFACTS_DIR / "feedback.csv"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ================================
# LOAD MODEL + METADATA
# ================================

if not CKPT_PATH.exists():
    raise RuntimeError(f"Checkpoint not found at {CKPT_PATH}")

ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)

n_users = ckpt["n_users"]
n_items = ckpt["n_items"]
gmf_dim = ckpt.get("gmf_dim", 32)
mlp_layers = (64, 32, 16, 8)

model = NeuMF(n_users, n_items, gmf_dim=gmf_dim, mlp_layers=mlp_layers).to(device)
model.load_state_dict(ckpt["state_dict"])
model.eval()

# Load mappings
maps = ckpt.get("maps", {})
user2idx = maps.get("user2idx", {})
item2idx = maps.get("item2idx", {})
idx2user = maps.get("idx2user", {})
idx2item = maps.get("idx2item", {})

# Load item metadata
item_df = pd.read_csv(ITEM_META_PATH)
item_df["item_idx"] = item_df["item_idx"].astype(int)
item_records = item_df.to_dict(orient="records")
item_by_idx = {int(r["item_idx"]): r for r in item_records}

# Load user map
user_df = pd.read_csv(USER_MAP_PATH)
user_df["user"] = user_df["user"].astype(str)
user_df["user_idx"] = user_df["user_idx"].astype(int)
user2idx = dict(zip(user_df["user"], user_df["user_idx"]))


# ================================
#  SAFE HELPERS
# ================================

def safe_str(x):
    if x is None:
        return ""
    try:
        if isinstance(x, float) and math.isnan(x):
            return ""
    except:
        pass
    return str(x)

def safe_int_or_none(x):
    try:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return None
        return int(x)
    except:
        return None


# ==================================================
#   NEW HELPER — MUST BE ABOVE recommend_by_item()
# ==================================================

def get_title_prefix(meta: dict) -> str:
    """
    Extract a rough 'franchise' prefix (e.g., 'Bakugan')
    """
    raw = (
        meta.get("title")
        or meta.get("title_english")
        or meta.get("title_japanese")
        or ""
    )
    title = str(raw).strip().lower()
    if not title:
        return ""

    # split by ':' or '-' to get main franchise name
    for sep in [":", "-"]:
        if sep in title:
            title = title.split(sep)[0].strip()
            break

    return title


# ================================
# BUILD ITEM EMBEDDINGS
# ================================

with torch.no_grad():
    gmf_item = model.gmf.item_embed.weight.to(device)
    mlp_item = model.mlp.item_embed.weight.to(device)
    item_emb_matrix = torch.cat([gmf_item, mlp_item], dim=1)
    item_emb_matrix = F.normalize(item_emb_matrix, dim=1)


# ================================
#  Pydantic models
# ================================

class Anime(BaseModel):
    mal_id: Optional[int] = None
    title: Optional[str] = None
    title_english: Optional[str] = None
    title_japanese: Optional[str] = None
    genres: Optional[str] = None
    url: Optional[str] = None           # MAL page
    image_url: Optional[str] = None     # NEW: poster image
    item_idx: int


class AnimeListResponse(BaseModel):
    total: int
    limit: int
    offset: int
    items: List[Anime]


class RecommendationItem(Anime):
    similarity: Optional[float] = None
    final_score: Optional[float] = None


class RecommendResponse(BaseModel):
    query: str
    anchor: dict
    items: List[RecommendationItem]


class FeedbackIn(BaseModel):
    user_id: str
    rating: int
    recommended_items: List[int]
    top_k: int
    comment: Optional[str] = None

    @field_validator("rating")
    def check_rating(cls, v):
        if not (1 <= v <= 5):
            raise ValueError("rating must be from 1 to 5")
        return v


# ================================
#  FASTAPI APP
# ================================

app = FastAPI(title="Anime Recommendation API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"status": "ok", "message": "Anime recommender backend is running"}


# ================================
#  /api/anime (catalogue)
# ================================

@app.get("/api/anime", response_model=AnimeListResponse)
def get_anime(limit: int = 20, offset: int = 0):
    """
    Return paginated anime list.

    We convert NaN / None values to safe types before creating the Anime model
    to avoid Pydantic validation errors (e.g. mal_id = NaN).
    """
    total = len(item_records)
    if offset < 0:
        offset = 0
    if limit < 1:
        limit = 1

    end = min(offset + limit, total)
    slice_records = item_records[offset:end]

    items: List[Anime] = []
    for r in slice_records:
        item = Anime(
            item_idx=int(r["item_idx"]),
            mal_id=safe_int_or_none(r.get("mal_id")),
            title=(safe_str(r.get("title")) or None),
            title_english=(safe_str(r.get("title_english")) or None),
            title_japanese=(safe_str(r.get("title_japanese")) or None),
            genres=(safe_str(r.get("genres")) or None),
            url=(safe_str(r.get("url")) or None),
            image_url=(safe_str(r.get("image_url")) or None),
        )
        items.append(item)

    return AnimeListResponse(
        total=total,
        limit=limit,
        offset=offset,
        items=items,
    )

# ================================
#  /api/anime/{item_idx}
# ================================

@app.get("/api/anime/{item_idx}", response_model=Anime)
def get_anime_details(item_idx: int):
    row = item_by_idx.get(item_idx)
    if not row:
        raise HTTPException(404, "Anime not found")
    return Anime(**row)


# ================================
#  /api/search_anime
# ================================

@app.get("/api/search_anime")
def search_anime(query: str):
    """
    Simple, safe search over ALL anime.
    Returns at most 50 matches containing the query
    in title / English title / Japanese title.
    """
    try:
        q = (query or "").strip().lower()
        if not q:
            return {"results": []}

        results = []
        for r in item_records:
            t = safe_str(r.get("title")).lower()
            e = safe_str(r.get("title_english")).lower()
            j = safe_str(r.get("title_japanese")).lower()

            if q in t or q in e or q in j:
                results.append(
                    {
                        "item_idx": int(r["item_idx"]),
                        "title": safe_str(r.get("title")) or None,
                        "title_english": safe_str(r.get("title_english")) or None,
                        "title_japanese": safe_str(r.get("title_japanese")) or None,
                        "mal_id": safe_int_or_none(r.get("mal_id")),
                        "genres": safe_str(r.get("genres")) or None,
                        "url": safe_str(r.get("url")) or None,
                    }
                )

            if len(results) >= 50:
                break

        return {"results": results}
    except Exception as e:
        # Log to server console and return clean 500
        print("ERROR in /api/search_anime:", e)
        raise HTTPException(status_code=500, detail=f"Search failed: {e}")


def get_title_keys(meta: dict):
    """
    Return two strings:
      - prefix: main series name before ':' / '-' etc.
      - head: first non-trivial word (e.g., 'bakugan', 'naruto').

    We use these to detect 'same franchise' between anchor and candidate.
    """
    raw = (
        meta.get("title")
        or meta.get("title_english")
        or meta.get("title_japanese")
        or ""
    )
    title = safe_str(raw).strip().lower()
    if not title:
        return "", ""

    # crude prefix: before ':' or '-' etc.
    for sep in [":", "-", "–", "—"]:
        if sep in title:
            title = title.split(sep)[0].strip()
            break

    # head word: first non-trivial token
    stop = {"the", "a", "an", "and", "of"}
    tokens = [t for t in title.replace("’", "'").split() if t and t not in stop]
    head = tokens[0] if tokens else ""

    return title, head


# ================================
#  /api/recommend_by_item  (with reranking)
# ================================

@app.get("/api/recommend_by_item")
def recommend_by_item(item_idx: int, top_k: int = 10):
    """
    Recommend similar anime given an item_idx.

    1) Use NeuMF item embeddings to get a large candidate pool.
    2) Rerank with:
         - franchise bonus (same prefix or same head word)
         - small genre bonus
    3) Hard rule: explicitly inject other titles from the same series
       (based on anchor head word in the title), so sequels show up.
    4) Return top_k after reranking.
    """

    if item_idx < 0 or item_idx >= n_items:
        raise HTTPException(status_code=404, detail="Unknown item_idx")

    anchor = item_by_idx.get(item_idx)
    if not anchor:
        raise HTTPException(status_code=404, detail="Anchor anime not found")

    # ----- anchor keys for reranking -----
    anchor_prefix, anchor_head = get_title_keys(anchor)
    anchor_genres = {
        g.strip().lower()
        for g in safe_str(anchor.get("genres")).split(",")
        if g.strip()
    }

    # ----- base similarity from model -----
    with torch.no_grad():
        target = item_emb_matrix[item_idx]           # [D]
        sims = torch.matmul(item_emb_matrix, target) # [n_items]
        sims = torch.nan_to_num(sims, nan=-1e9)

    # don't recommend itself
    sims[item_idx] = -1e9

    # big candidate pool from CF
    pool_k = min(max(top_k * 5, 50), n_items)
    top_scores, top_indices = torch.topk(sims, k=pool_k)

    recs = []
    in_recs = set()  # track item_idx already added

    for score, idx in zip(top_scores.tolist(), top_indices.tolist()):
        idx = int(idx)
        meta = item_by_idx.get(idx)
        if not meta:
            continue

        similarity = float(score)

        # ----- franchise keys for candidate -----
        cand_prefix, cand_head = get_title_keys(meta)

        same_prefix = anchor_prefix and cand_prefix and (anchor_prefix == cand_prefix)
        same_head = anchor_head and cand_head and (anchor_head == cand_head)

        # franchise bonus:
        franchise_bonus = 0.0
        if same_prefix:
            franchise_bonus = 0.14
        elif same_head:
            franchise_bonus = 0.08

        # ----- genre bonus -----
        cand_genres = {
            g.strip().lower()
            for g in safe_str(meta.get("genres")).split(",")
            if g.strip()
        }
        shared = anchor_genres & cand_genres
        genre_bonus = min(0.02 * len(shared), 0.06) if shared else 0.0

        final_score = similarity + franchise_bonus + genre_bonus

        rec = {
            "item_idx": idx,
            "mal_id": safe_int_or_none(meta.get("mal_id")),
            "title": safe_str(meta.get("title")),
            "title_english": safe_str(meta.get("title_english")),
            "title_japanese": safe_str(meta.get("title_japanese")),
            "genres": safe_str(meta.get("genres")),
            "url": safe_str(meta.get("url")),
            "image_url": safe_str(meta.get("image_url")),
            "similarity": similarity,
            "final_score": final_score,
        }
        recs.append(rec)
        in_recs.add(idx)

    # ---------------------------------------------------
    # HARD FRANCHISE RULE: inject all "same series" items
    # ---------------------------------------------------

    # use anchor_head as main series token (e.g. "bakugan", "naruto")
    series_token = anchor_head or anchor_prefix
    series_token = series_token.lower().strip() if series_token else ""

    if series_token:
        # current max score to place series items near the top
        max_final = max((r["final_score"] for r in recs), default=0.0)
        injected_bonus = max_final + 0.05  # put series items above others

        for r in item_records:
            idx = int(r["item_idx"])
            if idx == item_idx:
                continue
            if idx in in_recs:
                continue

            # combine titles and check if series token appears anywhere
            combined_title = (
                safe_str(r.get("title")).lower()
                + " "
                + safe_str(r.get("title_english")).lower()
                + " "
                + safe_str(r.get("title_japanese")).lower()
            )

            if series_token and series_token in combined_title:
                # base similarity if we have it, else 0
                similarity = float(sims[idx].item()) if 0 <= idx < sims.numel() else 0.0

                recs.append({
                    "item_idx": idx,
                    "mal_id": safe_int_or_none(r.get("mal_id")),
                    "title": safe_str(r.get("title")),
                    "title_english": safe_str(r.get("title_english")),
                    "title_japanese": safe_str(r.get("title_japanese")),
                    "genres": safe_str(r.get("genres")),
                    "url": safe_str(r.get("url")),
                    "similarity": similarity,
                    # force them near top with an injected score
                    "final_score": injected_bonus,
                })
                in_recs.add(idx)

    # final sort & trim
    recs.sort(key=lambda x: x["final_score"], reverse=True)
    recs = recs[: min(top_k, len(recs))]

    return {"item_idx": item_idx, "items": recs}

# ================================
#  /api/recommend_by_title
# ================================

@app.get("/api/recommend_by_title")
def recommend_by_title(query: str, top_k: int = 10):
    search = search_anime(query)
    results = search.get("results", [])

    if not results:
        raise HTTPException(404, "No anime found for that title")

    anchor = results[0]
    idx = int(anchor["item_idx"])

    recs = recommend_by_item(idx, top_k)

    return {
        "query": query,
        "anchor": anchor,
        "items": recs["items"]
    }


# ================================
#  /api/feedback
# ================================

@app.post("/api/feedback")
def submit_feedback(payload: FeedbackIn):
    FEEDBACK_PATH.parent.mkdir(exist_ok=True, parents=True)

    file_exists = FEEDBACK_PATH.exists()

    row = {
        "timestamp_utc": datetime.utcnow().isoformat(),
        "user_id": payload.user_id,
        "rating": payload.rating,
        "recommended_items": json.dumps(payload.recommended_items),
        "top_k": payload.top_k,
        "comment": payload.comment or "",
    }

    with FEEDBACK_PATH.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            w.writeheader()
        w.writerow(row)

    return {"status": "ok", "message": "Thank you for your feedback!"}

@app.get("/api/feedback_download")
def download_feedback():
    if not FEEDBACK_PATH.exists():
        raise HTTPException(404, "No feedback collected yet.")

    return FileResponse(
        FEEDBACK_PATH,
        media_type="text/csv",
        filename="feedback.csv"
    )


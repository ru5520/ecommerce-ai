"""商品库与 CLIP 索引（支持磁盘缓存与批量编码）。"""

import json
import os
import threading

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CSV_PATH = DATA_DIR / "products.csv"
INDEX_DIR = DATA_DIR / "clip_index"
INDEX_META = INDEX_DIR / "meta.json"
INDEX_EMB = INDEX_DIR / "embeddings.npy"
INDEX_IDS = INDEX_DIR / "product_ids.npy"
ENCODE_BATCH = 16

_DEFAULT_SNAPSHOT = (
    Path.home()
    / ".cache/huggingface/hub/models--openai--clip-vit-base-patch32/snapshots/3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"
)

_lock = threading.Lock()
_ready = False
_init_error: str | None = None
model = None
processor = None
products: pd.DataFrame | None = None
index = None


def _resolve_model_path() -> str:
    if path := os.environ.get("CLIP_MODEL_PATH"):
        return path
    if _DEFAULT_SNAPSHOT.is_dir():
        return str(_DEFAULT_SNAPSHOT)
    return "openai/clip-vit-base-patch32"


def is_ready() -> bool:
    return _ready


def init_error() -> str | None:
    return _init_error


def _csv_fingerprint() -> dict:
    st = CSV_PATH.stat()
    return {"mtime": st.st_mtime, "size": st.st_size}


def _cache_valid(meta: dict, ids: list[int], emb: np.ndarray) -> bool:
    if meta.get("model_path") != _resolve_model_path():
        return False
    fp = _csv_fingerprint()
    if meta.get("csv_mtime") != fp["mtime"] or meta.get("csv_size") != fp["size"]:
        return False
    if meta.get("count") != len(ids) or emb.shape[0] != len(ids):
        return False
    return True


def _load_cached_index() -> bool:
    global products, index
    if not INDEX_META.is_file() or not INDEX_EMB.is_file() or not INDEX_IDS.is_file():
        return False
    try:
        meta = json.loads(INDEX_META.read_text(encoding="utf-8"))
        ids = np.load(INDEX_IDS).astype(int).tolist()
        emb = np.load(INDEX_EMB)
        if not _cache_valid(meta, ids, emb):
            return False

        from api.product_admin import ensure_schema

        ensure_schema()
        all_products = pd.read_csv(CSV_PATH)
        if "visible" in all_products.columns:
            all_products = all_products[all_products["visible"].astype(int) == 1]
        all_products = all_products.reset_index(drop=True)
        id_to_row = {int(r["id"]): r for _, r in all_products.iterrows()}
        rows = []
        for pid in ids:
            if pid not in id_to_row:
                return False
            rows.append(id_to_row[pid])
        products = pd.DataFrame(rows).reset_index(drop=True)

        import faiss

        dimension = emb.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(emb.astype("float32"))
        print(f"Loaded CLIP index from cache ({len(ids)} products).")
        return True
    except Exception:
        return False


def _save_index_cache(ids: list[int], emb: np.ndarray) -> None:
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    np.save(INDEX_EMB, emb.astype("float32"))
    np.save(INDEX_IDS, np.array(ids, dtype=np.int32))
    fp = _csv_fingerprint()
    meta = {
        "model_path": _resolve_model_path(),
        "csv_mtime": fp["mtime"],
        "csv_size": fp["size"],
        "count": len(ids),
        "dim": int(emb.shape[1]),
    }
    INDEX_META.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def ensure_ready() -> None:
    global _ready, _init_error, model, processor, products, index
    if _ready:
        return
    with _lock:
        if _ready:
            return
        try:
            _ensure_model()
            if not _load_cached_index():
                _build_index()
            _ready = True
            _init_error = None
        except Exception as exc:
            _init_error = str(exc)
            raise


def rebuild_index() -> None:
    """商品变更后重建 FAISS 索引（复用已加载的 CLIP 模型）。"""
    global _ready, _init_error
    with _lock:
        try:
            _ensure_model()
            _build_index(force=True)
            _ready = True
            _init_error = None
        except Exception as exc:
            _init_error = str(exc)
            _ready = False
            raise


def _ensure_model() -> None:
    global model, processor
    if model is not None and processor is not None:
        return

    import torch
    from transformers import CLIPModel, CLIPProcessor

    model_path = _resolve_model_path()
    print(f"Loading CLIP model from {model_path}...")
    model = CLIPModel.from_pretrained(model_path)
    model.eval()
    try:
        processor = CLIPProcessor.from_pretrained(model_path, use_fast=True)
    except Exception:
        processor = CLIPProcessor.from_pretrained(model_path)
    print("CLIP loaded.")


def _to_tensor(out):
    """兼容不同 transformers 版本的返回类型（Tensor / ModelOutput / dict）。"""
    import torch

    if isinstance(out, torch.Tensor):
        return out
    # ModelOutput / dict-like
    for attr in ("image_embeds", "text_embeds", "last_hidden_state", "pooler_output"):
        v = getattr(out, attr, None)
        if isinstance(v, torch.Tensor):
            return v
    if isinstance(out, dict):
        for v in out.values():
            if isinstance(v, torch.Tensor):
                return v
    if isinstance(out, (list, tuple)) and out and isinstance(out[0], torch.Tensor):
        return out[0]
    raise TypeError(f"Unsupported model output type: {type(out)}")


def _encode_images(images: list) -> np.ndarray:
    import torch

    vectors = []
    for i in range(0, len(images), ENCODE_BATCH):
        batch = images[i : i + ENCODE_BATCH]
        inputs = processor(images=batch, return_tensors="pt")
        with torch.inference_mode():
            features = _to_tensor(model.get_image_features(**inputs))
        for row in features:
            emb = row.detach().cpu().numpy()
            emb = emb / np.linalg.norm(emb)
            vectors.append(emb)
    return np.array(vectors).astype("float32")


def _build_index(force: bool = False) -> None:
    global products, index

    import faiss
    from PIL import Image

    from api.product_admin import ensure_schema

    if not force and _load_cached_index():
        return

    ensure_schema()
    all_products = pd.read_csv(CSV_PATH)
    if "visible" in all_products.columns:
        all_products = all_products[all_products["visible"].astype(int) == 1]
    products = all_products.reset_index(drop=True)
    print(f"Loaded {len(products)} visible products for AI index.")

    print("Encoding product images (batched)...")
    pil_images = []
    ids: list[int] = []
    for _, row in products.iterrows():
        image_path = BASE_DIR / row["image_path"]
        if not image_path.is_file():
            raise FileNotFoundError(f"Missing product image: {image_path}")
        image = Image.open(image_path).convert("RGB")
        image.thumbnail((224, 224), Image.Resampling.LANCZOS)
        pil_images.append(image)
        ids.append(int(row["id"]))

    image_embeddings = _encode_images(pil_images)
    dimension = image_embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(image_embeddings)
    _save_index_cache(ids, image_embeddings)
    print("FAISS index built.")


def product_to_dict(row, score: float) -> dict:
    return {
        "id": int(row["id"]),
        "name": str(row["name"]),
        "price": float(row["price"]),
        "category": str(row["category"]),
        "image_path": str(row["image_path"]),
        "description": str(row.get("description", "") or ""),
        "score": float(score),
    }


def search_by_embedding(query_embedding: np.ndarray, top_k: int = 5) -> list[dict]:
    ensure_ready()
    query_embedding = query_embedding / np.linalg.norm(query_embedding)
    query_embedding = np.array([query_embedding]).astype("float32")
    scores, indices = index.search(query_embedding, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue
        results.append(product_to_dict(products.iloc[idx], score))
    return results


def search_by_image(image, top_k: int = 5) -> list[dict]:
    ensure_ready()
    import torch
    from PIL import Image as PILImage

    if not isinstance(image, PILImage.Image):
        image = PILImage.fromarray(image).convert("RGB")
    image.thumbnail((224, 224), PILImage.Resampling.LANCZOS)
    inputs = processor(images=image, return_tensors="pt")
    with torch.inference_mode():
        features = _to_tensor(model.get_image_features(**inputs))
    return search_by_embedding(features[0].detach().cpu().numpy(), top_k)


def search_by_text(text: str, top_k: int = 5) -> list[dict]:
    ensure_ready()
    import torch

    inputs = processor(text=[text], return_tensors="pt", padding=True)
    with torch.inference_mode():
        features = _to_tensor(model.get_text_features(**inputs))
    return search_by_embedding(features[0].detach().cpu().numpy(), top_k)

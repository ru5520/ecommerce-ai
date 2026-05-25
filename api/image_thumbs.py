"""商品图缩略图缓存，减轻 Gradio 公网隧道传输体积。"""

from __future__ import annotations

import hashlib
from pathlib import Path

from PIL import Image

BASE_DIR = Path(__file__).resolve().parent.parent
THUMB_DIR = BASE_DIR / "data" / "thumbs"
MAX_EDGE = 480
JPEG_QUALITY = 82


def _source_mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def _thumb_file(source: Path) -> Path:
    key = hashlib.md5(str(source.resolve()).encode(), usedforsecurity=False).hexdigest()[:16]
    return THUMB_DIR / f"{key}.jpg"


def ensure_thumb(image_path: str | Path) -> str:
    """返回缩略图路径（不存在则生成）。"""
    source = Path(image_path)
    if not source.is_absolute():
        source = BASE_DIR / source
    if not source.is_file():
        return str(source)

    THUMB_DIR.mkdir(parents=True, exist_ok=True)
    dest = _thumb_file(source)
    src_mtime = _source_mtime(source)
    if dest.is_file() and _source_mtime(dest) >= src_mtime:
        return str(dest)

    with Image.open(source) as img:
        img = img.convert("RGB")
        img.thumbnail((MAX_EDGE, MAX_EDGE), Image.Resampling.LANCZOS)
        img.save(dest, "JPEG", quality=JPEG_QUALITY, optimize=True)
    return str(dest)


def display_image_path(image_path: str) -> str:
    """商城列表 / 推荐画廊用缩略图；详情仍可用原图。"""
    try:
        return ensure_thumb(image_path)
    except Exception:
        p = Path(image_path)
        return str(p if p.is_absolute() else BASE_DIR / image_path)


def warm_thumbs(image_paths: list[str]) -> int:
    """预生成缩略图，返回成功数量。"""
    n = 0
    for path in image_paths:
        try:
            ensure_thumb(path)
            n += 1
        except Exception:
            pass
    return n

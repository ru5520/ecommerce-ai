from fastapi import APIRouter, File, UploadFile
from pydantic import BaseModel
from PIL import Image

from api.catalog import search_by_image, search_by_text
from api.llm_agent import rewrite_query

router = APIRouter()


class TextQuery(BaseModel):
    query: str
    top_k: int = 5
    use_llm: bool = True


def _response(results: list[dict], rewritten: str | None = None) -> dict:
    payload = {
        "top_match": results[0] if results else None,
        "all_results": results,
    }
    if rewritten is not None:
        payload["rewritten_query"] = rewritten
    return payload


@router.post("/search/image")
async def search_image(file: UploadFile = File(...)):
    image = Image.open(file.file).convert("RGB")
    results = search_by_image(image, top_k=5)
    return _response(results)


@router.post("/search/text")
async def search_text(body: TextQuery):
    results = search_by_text(body.query, top_k=body.top_k)
    return _response(results)


@router.post("/recommend")
async def recommend(body: TextQuery):
    query = body.query.strip()
    rewritten = query

    if body.use_llm:
        try:
            rewritten = rewrite_query(query)
        except Exception:
            pass

    results = search_by_text(rewritten, top_k=body.top_k)
    return _response(results, rewritten=rewritten)

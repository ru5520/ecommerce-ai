from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.image_search import router as search_router

app = FastAPI(title="AI E-commerce API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(search_router)


@app.get("/")
def root():
    return {
        "message": "AI E-commerce API",
        "endpoints": {
            "image_search": "POST /search/image",
            "text_search": "POST /search/text",
            "recommend": "POST /recommend",
        },
    }

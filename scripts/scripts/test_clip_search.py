from fastapi import APIRouter, UploadFile, File
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch

router = APIRouter()

print("Loading CLIP model...")

model_name = r"C:\Users\49130\.cache\huggingface\hub\models--openai--clip-vit-base-patch32\snapshots\3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"

model = CLIPModel.from_pretrained(
    model_name,
    local_files_only=True,
    use_safetensors=False
)

processor = CLIPProcessor.from_pretrained(
    model_name,
    local_files_only=True
)

print("CLIP loaded.")

# 商品标签
products = [
    "gaming laptop",
    "student notebook",
    "desktop gaming pc",
    "business laptop",
    "macbook"
]


@router.post("/image-search")
async def image_search(file: UploadFile = File(...)):

    image = Image.open(file.file)

    inputs = processor(
        text=products,
        images=image,
        return_tensors="pt",
        padding=True
    )

    with torch.no_grad():
        outputs = model(**inputs)

    probs = outputs.logits_per_image.softmax(dim=1)

    results = []

    for product, score in zip(products, probs[0]):
        results.append({
            "product": product,
            "score": float(score)
        })

    results = sorted(results, key=lambda x: x["score"], reverse=True)

    return {
        "top_match": results[0],
        "all_results": results
    }
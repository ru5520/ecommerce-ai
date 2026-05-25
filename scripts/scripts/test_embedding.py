from modelscope import snapshot_download
from sentence_transformers import SentenceTransformer

print("Downloading model...")

model_dir = snapshot_download(
    'sentence-transformers/all-MiniLM-L6-v2'
)

print("Loading model...")

model = SentenceTransformer(model_dir)

texts = [
    "light gaming laptop",
    "student notebook computer",
    "wireless bluetooth headphones"
]

embeddings = model.encode(texts)

print("\nEmbedding shape:")
print(embeddings.shape)

print("\nFirst vector:")
print(embeddings[0][:10])
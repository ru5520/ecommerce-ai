from modelscope import snapshot_download
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# -------------------
# 商品数据
# -------------------

products = [
    {
        "title": "light gaming laptop",
        "description": "RTX4060 gaming laptop for students"
    },
    {
        "title": "student notebook computer",
        "description": "lightweight notebook for college students"
    },
    {
        "title": "wireless bluetooth headphones",
        "description": "noise cancelling headphones"
    },
    {
        "title": "professional office laptop",
        "description": "business office ultrabook"
    },
    {
        "title": "gaming desktop computer",
        "description": "high performance RTX gaming PC"
    }
]

# -------------------
# 加载 embedding 模型
# -------------------

model_dir = snapshot_download(
    'sentence-transformers/all-MiniLM-L6-v2'
)

model = SentenceTransformer(model_dir)

# -------------------
# embedding
# -------------------

texts = [
    p["title"] + " " + p["description"]
    for p in products
]

embeddings = model.encode(texts)

embeddings = np.array(embeddings).astype("float32")

# -------------------
# FAISS
# -------------------

index = faiss.IndexFlatL2(embeddings.shape[1])

index.add(embeddings)

# -------------------
# 用户查询
# -------------------

query = "best laptop for college gaming"

query_embedding = model.encode([query])
query_embedding = np.array(query_embedding).astype("float32")

# -------------------
# 搜索
# -------------------

D, I = index.search(query_embedding, 3)

print("\nTop AI Recommendations:\n")

retrieved_products = []

for idx in I[0]:

    product = products[idx]

    retrieved_products.append(product)

    print(f"Product: {product['title']}")
    print(f"Description: {product['description']}")
    print()

# -------------------
# 模拟 LLM 总结
# -------------------

print("AI Summary:\n")

summary = """
Recommended for college students:
- lightweight gaming laptop
- RTX performance
- portable for study and gaming

Best choice:
light gaming laptop
"""

print(summary)
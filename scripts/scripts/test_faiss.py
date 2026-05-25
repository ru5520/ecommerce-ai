from modelscope import snapshot_download
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# 1. 加载模型
model_dir = snapshot_download(
    'sentence-transformers/all-MiniLM-L6-v2'
)

model = SentenceTransformer(model_dir)

# 2. 商品数据
products = [
    "light gaming laptop",
    "student notebook computer",
    "wireless bluetooth headphones",
    "professional office laptop",
    "gaming desktop computer"
]

# 3. embedding
embeddings = model.encode(products)

# 4. 转 float32
embeddings = np.array(embeddings).astype("float32")

# 5. 创建 FAISS 索引
index = faiss.IndexFlatL2(embeddings.shape[1])

# 6. 加入向量
index.add(embeddings)

print(f"FAISS index total: {index.ntotal}")

# 7. 用户查询
query = "best laptop for college gaming"

query_embedding = model.encode([query])
query_embedding = np.array(query_embedding).astype("float32")

# 8. 搜索最相似
D, I = index.search(query_embedding, 3)

print("\nTop results:")

for idx in I[0]:
    print(products[idx])
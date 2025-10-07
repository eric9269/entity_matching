# infer.py
from sentence_transformers import SentenceTransformer
import numpy as np

# 載入訓練好的模型
model = SentenceTransformer("models/bm25_bge-large-zh-v1.5")

# 把這個值改成你在訓練時印出的 best_t
# BEST_T = 0.45
#BEST_T = 0.575
BEST_T = 0.65


def predict_match(e3, e4, threshold=BEST_T):
    v1 = model.encode([e3], normalize_embeddings=True)
    v2 = model.encode([e4], normalize_embeddings=True)
    sim = float((v1 @ v2.T)[0,0])   # cosine（已 normalize）
    return ("match" if sim >= threshold else "no match", sim)

if __name__ == "__main__":
    # 單一範例
    e3 = "華碩 ROG Phone 8 16G/256G 黑"
    e4 = "ASUS ROG Phone 8 16GB 256GB Black"
    print("單一範例:", predict_match(e3, e4))

    # 多組驗證：品牌一樣商品不一樣、品牌不一樣商品類型一樣、完全不同等
    test_pairs = [
        # 品牌一樣，商品不一樣
        ("華碩 ROG Phone 8 16G/256G 黑", "華碩 Zenfone 10 8G/128G 白"),
        ("Apple iPhone 15 Pro 256GB Titanium Gray", "Apple iPad Pro 11吋 256GB 銀"),
        # 品牌不一樣，商品類型一樣
        ("三星 Galaxy S24 Ultra 12G/512G 黑", "Apple iPhone 15 Pro 256GB Titanium Gray"),
        ("小米 14 8G/256G 白", "OPPO Find X7 16GB 512GB Blue"),
        # 品牌一樣，型號接近但規格不同
        ("Sony Xperia 1 V 12G/256G 綠", "Sony Xperia 1 V 16G/512G 黑"),
        # 完全不同
        ("華為 Mate 60 Pro 12G/512G 灰", "Apple iPad Pro 11吋 256GB 銀"),
        # 應該 match
        ("ASUS ROG Phone 8 16G/256G 黑", "華碩 ROG Phone 8 16GB 256GB Black"),
        # 應該 no match
        ("ASUS ROG Phone 8 16G/256G 黑", "Apple iPhone 15 Pro 256GB Titanium Gray"),
    ]
    print("\n多組驗證:")
    for idx, (e3, e4) in enumerate(test_pairs, 1):
        result, sim = predict_match(e3, e4)
        print(f"{idx}. {e3} <-> {e4} => {result}, sim={sim:.4f}")

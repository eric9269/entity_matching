# build_pairs_and_split.py
import pandas as pd, numpy as np

# 讀檔（自行改路徑）
leaf = pd.read_csv("new_data/all_leaf (5).csv")
root = pd.read_csv("new_data/all_root (5).csv")
product = pd.read_csv("new_data/all_products (3).csv")

print(f"原始資料量 - leaf: {len(leaf)}, root: {len(root)}, product: {len(product)}")

# 只取需要欄位
leaf = leaf[['sku','title','platform','connect','query']].rename(columns={'sku':'Sku','title':'LeafTitle','platform':'Platform','connect':'Connect','query':'Query'})
root = root[['sku','title','platform']].rename(columns={'sku':'RootSku','title':'RootTitle','platform':'Platform'})
product = product[['sku','title','platform','query']].rename(columns={'sku':'ProductSku','title':'ProductTitle','platform':'Platform','query':'Query'})

# 去重：所有欄位都一模一樣的重複資料
print("開始去重...")
leaf_before = len(leaf)
root_before = len(root)
product_before = len(product)

leaf = leaf.drop_duplicates().reset_index(drop=True)
root = root.drop_duplicates().reset_index(drop=True)
product = product.drop_duplicates().reset_index(drop=True)

print(f"去重後資料量 - leaf: {len(leaf)} (減少 {leaf_before - len(leaf)}), root: {len(root)} (減少 {root_before - len(root)}), product: {len(product)} (減少 {product_before - len(product)})")

# 建 root 索引
root_idx = root.set_index('RootSku')

# -------- 正樣本 --------
pos = []
for r in leaf.itertuples():
    if pd.isna(r.Connect): 
        continue
    if r.Connect in root_idx.index:
        rt = root_idx.loc[r.Connect]
        pos.append({
            'e1': r.LeafTitle,
            'e2': rt['RootTitle'],
            'label': 1,
            'group_id': r.Connect,          # 以 rootSku 當群組，避免洩漏
            'platform': r.Platform,
            'query': r.Query
        })
df_pos = pd.DataFrame(pos).dropna()


# -------- 負樣本（基於 query 分群，用 BM25 挑選相似負樣本）---------

# BM25 挑選最相似負樣本 - 改為基於 query 分群
from rank_bm25 import BM25Okapi

neg_rows = []
neg_per_pos = 3

for row in df_pos.itertuples():
    query_type = row.query
    
    # 第一步：已知 e1，先找出在 all_leaf 中相同 connect 的所有 sku
    # 找到 e1 對應的 connect (即 group_id)
    target_connect = row.group_id
    
    # 找出所有與 e1 具有相同 connect 的 leaf sku
    same_connect_leaf_skus = set(leaf[leaf['Connect'] == target_connect]['Sku'].tolist())
    
    # 從 all_products 中找出相同 query 的商品（不限平台）
    same_query_products = product[product['Query'] == query_type].reset_index(drop=True)
    
    if len(same_query_products) < 2:
        # 如果還是太少，跳過這個正樣本
        print(f"Warning: Not enough products for query '{query_type}', skipping...")
        continue
    
    # 第二步：刪減掉剛剛記錄到的 sku，不能納入 BM25 篩選
    # 排除條件：1) 與e1商品名稱相同 2) ProductSku 在相同 connect 的 leaf sku 中
    filtered_products = same_query_products[
        (same_query_products['ProductTitle'] != row.e1) &  # 排除相同商品名稱
        (~same_query_products['ProductSku'].isin(same_connect_leaf_skus))  # 排除相同 connect 的 sku
    ].reset_index(drop=True)
    
    if len(filtered_products) < neg_per_pos:
        # 如果過濾後商品不足，就用全部
        neg_count = len(filtered_products)
    else:
        neg_count = neg_per_pos
    
    if neg_count == 0:
        continue
    
    # 從候選商品中用 BM25 挑選最相似的
    candidate_titles = filtered_products['ProductTitle'].tolist()
    tokenized_corpus = [title.split() for title in candidate_titles]
    bm25 = BM25Okapi(tokenized_corpus)
    query_tokens = row.e1.split()
    scores = bm25.get_scores(query_tokens)
    
    # 取分數最高的 neg_count 個
    top_idx = np.argsort(scores)[::-1][:neg_count]
    
    for idx in top_idx:
        cand = filtered_products.iloc[idx]
        neg_rows.append({
            'e1': row.e1,
            'e2': cand['ProductTitle'],
            'label': 0,
            'group_id': row.group_id,
            'platform': row.platform,
            'query': query_type
        })

df_neg = pd.DataFrame(neg_rows)

# 合併
df_all = pd.concat([df_pos, df_neg], ignore_index=True)
df_all = df_all.sample(frac=1.0, random_state=42).reset_index(drop=True)
df_all[['e1','e2','label','group_id','query']].to_csv("new_data/pairs_all_bm25.csv", index=False)

print("pairs_all_bm25.csv saved",
      "total=", len(df_all),
      "pos=", (df_all.label==1).sum(),
      "neg=", (df_all.label==0).sum())
print("Query distribution in positive samples:")
print(df_pos['query'].value_counts())

# -------- 印出十組範例（1正樣本+3負樣本）---------
print("\n=== 十組範例 ===")
unique_groups = df_pos['group_id'].unique()[:5]  # 取前5組
for i, group_id in enumerate(unique_groups, 1):
    print(f"\n--- 第 {i} 組 (group_id: {group_id}) ---")
    
    # 找正樣本
    pos_sample = df_pos[df_pos['group_id'] == group_id].iloc[0]
    print(f"正樣本 (label=1):")
    print(f"  Query: {pos_sample['query']}")
    print(f"  e1: {pos_sample['e1']}")
    print(f"  e2: {pos_sample['e2']}")
    
    # 找對應的負樣本
    neg_samples = df_neg[df_neg['group_id'] == group_id]
    print(f"\n負樣本 (label=0):")
    for j, neg_sample in neg_samples.iterrows():
        print(f"  e1: {neg_sample['e1']}")
        print(f"  e2: {neg_sample['e2']}")
        print()

# -------- 依 group_id 做 80/20 切分（避免洩漏）---------
groups = df_all['group_id'].drop_duplicates().sample(frac=1.0, random_state=42).tolist()
split = int(0.8*len(groups))
train_groups = set(groups[:split])
test_groups  = set(groups[split:])

train_df = df_all[df_all['group_id'].isin(train_groups)].reset_index(drop=True)
test_df  = df_all[df_all['group_id'].isin(test_groups)].reset_index(drop=True)

train_df.to_csv("new_data/pairs_train_bm25.csv", index=False)
test_df.to_csv("new_data/pairs_test_bm25.csv", index=False)
print("train:", train_df.shape, "test:", test_df.shape)

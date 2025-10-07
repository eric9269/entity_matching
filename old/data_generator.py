# build_pairs_and_split.py
import pandas as pd, numpy as np

# 讀檔（自行改路徑）
leaf = pd.read_csv("all_leaf.csv")
root = pd.read_csv("all_root.csv")

# 只取需要欄位
leaf = leaf[['Sku','Title','Platform','Connect']].rename(columns={'Title':'LeafTitle'})
root = root[['Sku','Title','Platform']].rename(columns={'Sku':'RootSku','Title':'RootTitle'})

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
            'platform': r.Platform
        })
df_pos = pd.DataFrame(pos).dropna()

# 若你確定正樣本大約 2000，可用下行限制大小（不需要就註解掉）
# df_pos = df_pos.sample(n=2000, random_state=42)  # <-- 可選

# -------- 負樣本（同平台隨機配，3 倍）---------
rng = np.random.default_rng(42)
root_by_plat = {p: sub.reset_index(drop=True) for p, sub in root.groupby('Platform')}
neg_rows = []
neg_per_pos = 3

for row in df_pos.itertuples():
    plat = row.platform
    pool = root_by_plat.get(plat, None)
    if pool is None or len(pool) < 2:
        # 若該平台 pool 太小，跨平台兜底
        any_plat = rng.choice(list(root_by_plat.keys()))
        pool = root_by_plat[any_plat]
    cnt = 0
    # 取 3 個不同 root 且 != 真實 group_id
    tried = set()
    while cnt < neg_per_pos and len(tried) < 50:
        idx = int(rng.integers(0, len(pool)))
        cand = pool.iloc[idx]
        tried.add(idx)
        if cand['RootSku'] == row.group_id:
            continue
        neg_rows.append({
            'e1': row.e1,
            'e2': cand['RootTitle'],
            'label': 0,
            'group_id': row.group_id,   # 仍用真實實體當群組
            'platform': plat
        })
        cnt += 1

df_neg = pd.DataFrame(neg_rows)

# 合併
df_all = pd.concat([df_pos, df_neg], ignore_index=True)
df_all = df_all.sample(frac=1.0, random_state=42).reset_index(drop=True)
df_all[['e1','e2','label','group_id']].to_csv("data/pairs_all.csv", index=False)

print("pairs_all.csv saved",
      "total=", len(df_all),
      "pos=", (df_all.label==1).sum(),
      "neg=", (df_all.label==0).sum())

# -------- 依 group_id 做 80/20 切分（避免洩漏）---------
groups = df_all['group_id'].drop_duplicates().sample(frac=1.0, random_state=42).tolist()
split = int(0.8*len(groups))
train_groups = set(groups[:split])
test_groups  = set(groups[split:])

train_df = df_all[df_all['group_id'].isin(train_groups)].reset_index(drop=True)
test_df  = df_all[df_all['group_id'].isin(test_groups)].reset_index(drop=True)

train_df.to_csv("data/pairs_train.csv", index=False)
test_df.to_csv("data/pairs_test.csv", index=False)
print("train:", train_df.shape, "test:", test_df.shape)

# train_level2_biencoder.py
import pandas as pd, numpy as np
from sentence_transformers import SentenceTransformer, InputExample, losses, evaluation, util
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, classification_report, average_precision_score

# 讀入切好的檔
train_df = pd.read_csv("old/bm25_data/pairs_train_bm25.csv")
test_df  = pd.read_csv("old/bm25_data/pairs_test_bm25.csv")

# 再從 train 切出 valid（以 group_id 切）
groups = train_df['group_id'].drop_duplicates().sample(frac=1.0, random_state=42).tolist()
split = int(0.85*len(groups))  # 85% train, 15% valid（在 80% 大集中再切）
tr_groups = set(groups[:split])
va_groups = set(groups[split:])

tr = train_df[train_df['group_id'].isin(tr_groups)]
va = train_df[train_df['group_id'].isin(va_groups)]

train_samples = [InputExample(texts=[r.e1, r.e2], label=float(r.label)) for r in tr.itertuples()]
valid_evaluator = evaluation.BinaryClassificationEvaluator(
    sentences1=va['e1'].tolist(),
    sentences2=va['e2'].tolist(),
    labels=va['label'].astype(float).tolist()
)

# 可換成 "uer/sbert-base-chinese-nli"
# model = SentenceTransformer("shibing624/text2vec-base-chinese")
model = SentenceTransformer("BAAI/bge-large-zh-v1.5")

loader = DataLoader(train_samples, shuffle=True, batch_size=64)
loss_fn = losses.CosineSimilarityLoss(model)

model.fit(
    train_objectives=[(loader, loss_fn)],
    epochs=15,
    warmup_steps=int(0.1*len(loader)),
    evaluator=valid_evaluator,
    evaluation_steps=200,
    output_path="old/models/(15)bm25_bge-large-zh-v1.5"
)

# ---- 在 valid 找最佳 threshold（以 F1 最佳）----
def find_best_t(mdl, s1, s2, y_true):
    e1 = mdl.encode(s1, batch_size=128, convert_to_tensor=True, normalize_embeddings=True)
    e2 = mdl.encode(s2, batch_size=128, convert_to_tensor=True, normalize_embeddings=True)
    sims = util.cos_sim(e1, e2).diagonal().cpu().numpy()
    best_t, best_f1 = 0.5, -1
    for t in np.linspace(0.2, 0.95, 31):
        f1 = f1_score(y_true, (sims>=t).astype(int))
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t

best_t = find_best_t(model, va['e1'].tolist(), va['e2'].tolist(), va['label'].to_numpy().astype(int))
print("Best threshold on valid =", round(best_t,3))

# ---- 在測試集評估 ----
e1 = model.encode(test_df['e1'].tolist(), batch_size=128, convert_to_tensor=True, normalize_embeddings=True)
e2 = model.encode(test_df['e2'].tolist(), batch_size=128, convert_to_tensor=True, normalize_embeddings=True)
sims = util.cos_sim(e1, e2).diagonal().cpu().numpy()
y_true = test_df['label'].to_numpy().astype(int)

y_pred = (sims >= best_t).astype(int)
print("PR-AUC:", round(average_precision_score(y_true, sims), 4))
print(classification_report(y_true, y_pred, digits=3))
print("Suggested threshold:", round(best_t,3))

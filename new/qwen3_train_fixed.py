# train_level2_biencoder.py
import pandas as pd, numpy as np
from sentence_transformers import SentenceTransformer, InputExample, losses, evaluation, util
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, classification_report, average_precision_score
from peft import LoraConfig, get_peft_model, TaskType
import torch
import gc

# 讀入切好的檔
train_df = pd.read_csv("new_data/pairs_train_bm25.csv")
test_df  = pd.read_csv("new_data/pairs_test_bm25.csv")

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

# 清理顯存
torch.cuda.empty_cache()
gc.collect()

# 載入 Qwen3-Embedding-4B 模型，使用壓縮維度
model = SentenceTransformer(
    "Qwen/Qwen3-Embedding-4B",
    model_kwargs={
        "device_map": "auto",
    },
    truncate_dim=768  # 大幅壓縮維度到 384
)

# 配置 LoRA 參數
lora_config = LoraConfig(
    task_type=TaskType.FEATURE_EXTRACTION,
    r=8,  # 進一步降低 LoRA rank 以節省顯存
    lora_alpha=8,  # 相應調整 alpha
    lora_dropout=0.1,  # LoRA dropout
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # 目標模組
    bias="none",
    inference_mode=False,
)

# 為模型的第一層（通常是 transformer）應用 LoRA
if hasattr(model[0], 'auto_model'):
    model[0].auto_model = get_peft_model(model[0].auto_model, lora_config)
    print("✅ LoRA 已成功應用到 Qwen3-Embedding-4B 模型 (truncate_dim=384)")
else:
    print("⚠️ 無法找到 auto_model 屬性，LoRA 配置可能需要調整")

# 再次清理顯存
torch.cuda.empty_cache()
gc.collect()

# 使用極小 batch size 控制顯存使用
loader = DataLoader(train_samples, shuffle=True, batch_size=4)  # 極小批次大小
loss_fn = losses.CosineSimilarityLoss(model)

model.fit(
    train_objectives=[(loader, loss_fn)],
    epochs=10,  # 減少訓練輪數
    warmup_steps=int(0.1*len(loader)),
    evaluator=valid_evaluator,
    evaluation_steps=1000,  # 增加評估間隔
    output_path="models/768-10-qwen3-embedding-4b-lora"
)

# ---- 在 valid 找最佳 threshold（以 F1 最佳）----
def find_best_t(mdl, s1, s2, y_true):
    # 降低編碼 batch size 以節省顯存
    e1 = mdl.encode(s1, batch_size=16, convert_to_tensor=True, normalize_embeddings=True)
    e2 = mdl.encode(s2, batch_size=16, convert_to_tensor=True, normalize_embeddings=True)
    sims = util.cos_sim(e1, e2).diagonal().cpu().numpy()
    
    # 清理顯存
    torch.cuda.empty_cache()
    gc.collect()
    
    best_t, best_f1 = 0.5, -1
    for t in np.linspace(0.2, 0.95, 31):
        f1 = f1_score(y_true, (sims>=t).astype(int))
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t

try:
    best_t = find_best_t(model, va['e1'].tolist(), va['e2'].tolist(), va['label'].to_numpy().astype(int))
    print("Best threshold on valid =", round(best_t,3))
except Exception as e:
    print(f"找最佳閾值時發生錯誤: {e}")
    print("使用預設閾值 0.5")
    best_t = 0.5

# 最終清理顯存
del model
torch.cuda.empty_cache()
gc.collect()
print("✅ 模型已釋放，顯存已清理")
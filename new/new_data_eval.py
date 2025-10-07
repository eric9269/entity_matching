import pandas as pd
from sentence_transformers import SentenceTransformer, util
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, average_precision_score

model = SentenceTransformer("models/13-qwen3-embedding-4b-lora")
test_df  = pd.read_csv("new_data/pairs_test_bm25.csv")
best_t = 0.275

# ---- 在測試集評估 ----
e1 = model.encode(test_df['e1'].tolist(), batch_size=128, convert_to_tensor=True, normalize_embeddings=True)
e2 = model.encode(test_df['e2'].tolist(), batch_size=128, convert_to_tensor=True, normalize_embeddings=True)
sims = util.cos_sim(e1, e2).diagonal().cpu().numpy()
y_true = test_df['label'].to_numpy().astype(int)

y_pred = (sims >= best_t).astype(int)
print("PR-AUC:", round(average_precision_score(y_true, sims), 4))
print(classification_report(y_true, y_pred, digits=3))
print("Suggested threshold:", round(best_t,3))

# === 記錄錯誤預測到 txt 檔 ===
wrong_predictions = []
for i, (true_label, pred_label, sim_score) in enumerate(zip(y_true, y_pred, sims)):
    if true_label != pred_label:
        row = test_df.iloc[i]
        error_type = "False Positive" if pred_label == 1 else "False Negative"
        wrong_predictions.append({
            'index': i,
            'e1': row['e1'], 
            'e2': row['e2'],
            'true_label': true_label,
            'pred_label': pred_label,
            'similarity': sim_score,
            'error_type': error_type,
            'group_id': row.get('group_id', 'N/A'),
            'query': row.get('query', 'N/A')
        })

# 寫入錯誤預測檔案
with open('error/13-qwen3-embedding-4b-lora', 'w', encoding='utf-8') as f:
    f.write(f"=== 錯誤預測分析報告 ===\n")
    f.write(f"模型: new/models/13-qwen3-embedding-4b-lora\n")
    f.write(f"閾值: {best_t}\n")
    f.write(f"總測試樣本: {len(y_true)}\n")
    f.write(f"錯誤預測數: {len(wrong_predictions)}\n")
    f.write(f"錯誤率: {len(wrong_predictions)/len(y_true):.3f}\n\n")
    
    # 統計錯誤類型
    fp_count = sum(1 for w in wrong_predictions if w['error_type'] == 'False Positive')
    fn_count = sum(1 for w in wrong_predictions if w['error_type'] == 'False Negative') 
    f.write(f"False Positive (誤判為匹配): {fp_count}\n")
    f.write(f"False Negative (漏判匹配): {fn_count}\n\n")
    
    # 分離 False Negative 和 False Positive
    false_negatives = [w for w in wrong_predictions if w['error_type'] == 'False Negative']
    false_positives = [w for w in wrong_predictions if w['error_type'] == 'False Positive']
    
    # 第一區塊：False Negative（實際匹配但預測不匹配）
    f.write("=" * 80 + "\n")
    f.write("區塊一：False Negative - 實際匹配但預測不匹配（漏判）\n")
    f.write("=" * 80 + "\n\n")
    
    if false_negatives:
        for j, error in enumerate(false_negatives, 1):
            f.write(f"漏判 #{j}\n")
            f.write(f"索引: {error['index']}\n")
            f.write(f"Query: {error['query']}\n") 
            f.write(f"Group ID: {error['group_id']}\n")
            f.write(f"e1: {error['e1']}\n")
            f.write(f"e2: {error['e2']}\n")
            f.write(f"實際標籤: {error['true_label']} (匹配)\n")
            f.write(f"預測標籤: {error['pred_label']} (不匹配)\n")
            f.write(f"相似度分數: {error['similarity']:.4f} (< {best_t})\n")
            f.write("-" * 60 + "\n\n")
    else:
        f.write("沒有 False Negative 錯誤\n\n")
    
    # 第二區塊：False Positive（實際不匹配但預測匹配）
    f.write("=" * 80 + "\n")
    f.write("區塊二：False Positive - 實際不匹配但預測匹配（誤判）\n") 
    f.write("=" * 80 + "\n\n")
    
    if false_positives:
        for j, error in enumerate(false_positives, 1):
            f.write(f"誤判 #{j}\n")
            f.write(f"索引: {error['index']}\n")
            f.write(f"Query: {error['query']}\n")
            f.write(f"Group ID: {error['group_id']}\n") 
            f.write(f"e1: {error['e1']}\n")
            f.write(f"e2: {error['e2']}\n")
            f.write(f"實際標籤: {error['true_label']} (不匹配)\n")
            f.write(f"預測標籤: {error['pred_label']} (匹配)\n")
            f.write(f"相似度分數: {error['similarity']:.4f} (>= {best_t})\n")
            f.write("-" * 60 + "\n\n")
    else:
        f.write("沒有 False Positive 錯誤\n\n")

print(f"\n錯誤預測已記錄至 error_predictions.txt")
print(f"共 {len(wrong_predictions)} 個錯誤預測 (錯誤率: {len(wrong_predictions)/len(y_true):.3f})")


print("\n=== 測試集五組範例 ===")
sample_test = test_df.sample(n=5, random_state=42)  # 隨機選5組
for i, row in enumerate(sample_test.itertuples(), 1):
    # 計算模型預測的相似度
    emb1 = model.encode([row.e1], normalize_embeddings=True)
    emb2 = model.encode([row.e2], normalize_embeddings=True)
    sim = float((emb1 @ emb2.T)[0,0])
    
    # 根據最佳閾值預測
    prediction = "match" if sim >= best_t else "no match"
    actual = "match" if row.label == 1 else "no match"
    result = "✅" if prediction == actual else "❌"
    
    print(f"\n--- 第 {i} 組 ---")
    print(f"e1: {row.e1}")
    print(f"e2: {row.e2}")
    print(f"實際標籤: {actual}")
    print(f"模型預測: {prediction} (相似度: {sim:.4f}) {result}")

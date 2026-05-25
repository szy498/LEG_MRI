import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.stats import spearmanr
import os
import matplotlib.pyplot as plt
import seaborn as sns

# ===================== 基本参数 =====================
folder_name = "fe"

radiomics_file = f"E:/US-MRI/{folder_name}/radiomics.xlsx"
original_file = f"E:/US-MRI/{folder_name}/1.xlsx"

patient_id_col = "Patient_ID"
target_var = "Diabetes_Status"   # 仅用于分层抽样，不参与特征筛选

threshold_corr = 0.9
n_splits = 10
test_size = 0.3
random_base = 42

split_root = f"E:/US-MRI/{folder_name}/split_results"
output_dir = f"E:/US-MRI/{folder_name}/redundancy_stability_radom2/"
os.makedirs(split_root, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

np.random.seed(42)

# ===================== 读取数据 =====================
df_radiomics = pd.read_excel(radiomics_file, engine='openpyxl')
features = df_radiomics.columns.drop(patient_id_col)

df_original = pd.read_excel(original_file, engine='openpyxl')

# 初始化计数字典
feature_counts = {f: 0 for f in features}

# ===================== 主循环 =====================
for run_id in range(1, n_splits + 1):
    print(f"\n====== Run {run_id} ======")

    run_dir = os.path.join(split_root, f"run_{run_id}")
    os.makedirs(run_dir, exist_ok=True)

    train_file = os.path.join(run_dir, "train_set.xlsx")
    test_file = os.path.join(run_dir, "test_set.xlsx")

    # ---------- Step 1: 划分训练/测试集 ----------
    if os.path.exists(train_file) and os.path.exists(test_file):
        print(f"📁 读取已有划分: run_{run_id}")
        train_df = pd.read_excel(train_file, engine='openpyxl')
        test_df = pd.read_excel(test_file, engine='openpyxl')
    else:
        print(f"🆕 创建新划分: run_{run_id}")
        train_df, test_df = train_test_split(
            df_original,
            test_size=test_size,
            stratify=df_original[target_var],
            random_state=random_base + run_id
        )
        train_df.to_excel(train_file, index=False, engine='openpyxl')
        test_df.to_excel(test_file, index=False, engine='openpyxl')

    # ---------- Step 2: 构建训练集特征矩阵 ----------
    df_radiomics_train = df_radiomics[
        df_radiomics[patient_id_col].isin(train_df[patient_id_col])
    ].copy()

    X_train = df_radiomics_train[features]

    # ---------- Step 3: 仅计算“特征-特征”相关性 ----------
    print("📊 计算特征间 Spearman 相关矩阵（无监督）...")
    corr_matrix = X_train.corr(method='spearman').abs()

    # ---------- Step 4: 冗余特征无监督剔除逻辑 ----------
    to_drop = set()
    mean_corr = corr_matrix.mean(axis=1)

    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            f1, f2 = features[i], features[j]
            if corr_matrix.loc[f1, f2] > threshold_corr:
                # 删除“平均相关性更高”的那个（无监督）
                if mean_corr[f1] >= mean_corr[f2]:
                    to_drop.add(f1)
                else:
                    to_drop.add(f2)

    selected_features = [f for f in features if f not in to_drop]

    # ---------- Step 5: 统计稳定性 ----------
    for f in selected_features:
        feature_counts[f] += 1

    # ---------- Step 6: 保存单次结果 ----------
    pd.DataFrame({"Selected_Features": selected_features}).to_excel(
        os.path.join(output_dir, f"selected_features_run_{run_id}.xlsx"),
        index=False
    )

    print(f"✅ Run {run_id}: 保留特征数 = {len(selected_features)}")

# ===================== Step 7: 汇总稳定性 =====================
stability_df = pd.DataFrame({
    "Feature": list(feature_counts.keys()),
    "Appearance": list(feature_counts.values()),
    "Frequency": [v / n_splits for v in feature_counts.values()]
}).sort_values("Frequency", ascending=False)

stability_file = os.path.join(output_dir, "feature_stability_summary.xlsx")
stability_df.to_excel(stability_file, index=False)

print("\n✅ 稳定性统计完成！")
print(f"📁 稳定性文件：{stability_file}")

# ===================== Step 8: 可视化 Top 特征 =====================
plt.figure(figsize=(12, 8))
top_features = stability_df.head(30)

sns.barplot(
    x="Frequency",
    y="Feature",
    data=top_features,
    palette="viridis"
)

plt.xlabel("Appearance Frequency (out of runs)")
plt.ylabel("Feature")
plt.title(f"Top {len(top_features)} Stable Radiomics Features")
plt.tight_layout()

fig_path = os.path.join(output_dir, "top_feature_stability.png")
plt.savefig(fig_path, dpi=300)
plt.show()

print("\n🎉 全部处理完成！")
print(f"📁 划分结果目录：{split_root}")
print(f"📁 输出结果目录：{output_dir}")

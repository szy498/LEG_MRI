#################影像组学特征的
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RepeatedStratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import matplotlib
import shap
matplotlib.use('Agg')  # 不使用窗口，直接保存图像
def plot_mean_roc(roc_list, label):
    mean_fpr = np.linspace(0, 1, 100)
    tprs = []
    aucs = []

    for y_true, y_prob in roc_list:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)

        tpr_interp = np.interp(mean_fpr, fpr, tpr)
        tpr_interp[0] = 0.0
        tprs.append(tpr_interp)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0

    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)

    plt.plot(
        mean_fpr,
        mean_tpr,
        linewidth=2,
        label=f"{label} (AUC = {mean_auc:.3f})"
    )
from sklearn.utils import resample
from sklearn.metrics import roc_curve, auc
def plot_mean_roc_bootstrap_custom(roc_list, label, color, alpha_ci=0.15, ci_level=0.95):
    """
    roc_list: list of (y_true, y_prob), each from ONE outer CV fold
    """

    mean_fpr = np.linspace(0, 1, 100)
    tprs = []
    aucs = []

    # ===== 1. 每个 CV fold 计算一条 ROC =====
    for y_true, y_prob in roc_list:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        fold_auc = auc(fpr, tpr)
        aucs.append(fold_auc)

        tpr_interp = np.interp(mean_fpr, fpr, tpr)
        tpr_interp[0] = 0.0
        tprs.append(tpr_interp)

    tprs = np.array(tprs)
    aucs = np.array(aucs)

    # ===== 2. mean ROC =====
    mean_tpr = tprs.mean(axis=0)
    mean_tpr[-1] = 1.0

    # ===== 3. 95% CI of ROC (CV-level percentile) =====
    lower_tpr = np.percentile(
        tprs, (1 - ci_level) / 2 * 100, axis=0
    )
    upper_tpr = np.percentile(
        tprs, (1 + ci_level) / 2 * 100, axis=0
    )

    # ===== 4. AUC = mean(Test_AUC) + 95% CI =====
    mean_auc = aucs.mean()
    lower_auc = np.percentile(
        aucs, (1 - ci_level) / 2 * 100
    )
    upper_auc = np.percentile(
        aucs, (1 + ci_level) / 2 * 100
    )

    # ===== 5. Plot =====
    plt.plot(
        mean_fpr,
        mean_tpr,
        lw=2,
        color=color,
        label=f"{label} (AUC = {mean_auc:.3f})"
    )
    plt.fill_between(
        mean_fpr,
        lower_tpr,
        upper_tpr,
        color=color,
        alpha=alpha_ci
    )

# =========================
# 参数配置
# =========================
target_var = "Diabetes_Status"
radiomics_feature_names = [


                        "Muscle_Fat_original_firstorder_InterquartileRange",
                        "Muscle_Fat_original_firstorder_Kurtosis",
                        "Muscle_Fat_original_firstorder_Mean",
                        "Muscle_Fat_original_firstorder_Median",
                        "Muscle_Fat_original_firstorder_Skewness",
                        "Muscle_Fat_original_firstorder_Variance",
                        "IMAT_Fat_original_firstorder_InterquartileRange",
                        "IMAT_Fat_original_firstorder_Kurtosis",
                        "IMAT_Fat_original_firstorder_Mean",
                        "IMAT_Fat_original_firstorder_Median",
                        "IMAT_Fat_original_firstorder_Skewness",
                        "IMAT_Fat_original_firstorder_Variance",
                        "Subcutaneous_Fat_Fat_original_firstorder_InterquartileRange",
                        "Subcutaneous_Fat_Fat_original_firstorder_Kurtosis",
                        "Subcutaneous_Fat_Fat_original_firstorder_Mean",
                        "Subcutaneous_Fat_Fat_original_firstorder_Median",
                        "Subcutaneous_Fat_Fat_original_firstorder_Skewness",
                        "Subcutaneous_Fat_Fat_original_firstorder_Variance",

                        "Muscle_original_shape_Flatness",
                        "Muscle_original_shape_Sphericity",
                        "IMAT_original_shape_Flatness",
                        "IMAT_original_shape_Sphericity",
                        "Subcutaneous_Fat_original_shape_Flatness",
                        "Subcutaneous_Fat_original_shape_Sphericity",

                        "Muscle_Fat_original_glcm_Imc1",
                        "Muscle_Water_original_glcm_SumEntropy",
                        "IMAT_Fat_original_glcm_ClusterShade",
                        "Subcutaneous_Fat_Fat_original_glcm_ClusterProminence",
                        "IMAT_Water_original_glcm_ClusterTendency",
                        "IMAT_Water_original_glcm_ClusterShade",
                        "IMAT_Fat_original_glcm_MaximumProbability",
                        "Muscle_Fat_original_glcm_MCC",
                        "Muscle_Fat_original_glcm_InverseVariance",
                        "IMAT_Water_original_glcm_MaximumProbability"
]

clinical_feature_names = ["BMI", "Age", "Sex"]

feature_sets = {
    "Radiomics_Only": radiomics_feature_names,
    "Radiomics_+_Clinical": radiomics_feature_names + clinical_feature_names
}

folder_name = "fe"
data_file = f"E:/US-MRI/{folder_name}/radiomics.xlsx"
# 标签文件（含 Diabetes_Status）
Status_file = f"E:/US-MRI/{folder_name}/1.xlsx"
id_col = "Patient_ID"
# =========================
# 读取数据
# =========================
df_feat = pd.read_excel(data_file, engine="openpyxl")
df_split = pd.read_excel(Status_file, engine="openpyxl")

df = pd.merge(
    df_split[[id_col, target_var] + clinical_feature_names],
    df_feat[[id_col] + radiomics_feature_names],
    on=id_col,
    how="inner"
)

# 安全校验（防止论文审稿人质疑）
missing_cols = [col for col in [target_var] + clinical_feature_names + radiomics_feature_names if col not in df.columns]
if missing_cols:
    raise ValueError(f"❌ Missing columns after merge: {missing_cols}")

print("✅ Merged dataset shape:", df.shape)

# =========================
# 严格 Nested CV 设置
# =========================
outer_cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=42)
inner_cv_fs = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=1)
inner_cv_tune = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=7)

# =========================
# 结果容器
# =========================
results = []
feature_stability = {k: Counter() for k in feature_sets.keys()}
# =========================
# ROC 容器（用于论文画 ROC）
# =========================
roc_storage = {
    "Radiomics_Only": {"LR": [], "RF": [], "XGB": []},
    "Radiomics_+_Clinical": {"LR": [], "RF": [], "XGB": []}
}
# 每个元素存 (y_test, y_prob)

# =========================
# 主循环
# =========================
# for set_name, features in feature_sets.items():
#     print(f"\n==== Feature Set: {set_name} ====")
#
#     X = df[features].values
#     y = df[target_var].values
X_radiomics = df[radiomics_feature_names].values
X_clinical = df[clinical_feature_names].values
y = df[target_var].values

fold_id = 1
shap_storage = {
    "Radiomics_Only": {"LR": [], "RF": [], "XGB": []},
    "Radiomics_+_Clinical": {"LR": [], "RF": [], "XGB": []}
}

for train_idx, test_idx in outer_cv.split(X_radiomics, y):
    print(f"Outer Fold {fold_id}")

    # 外层划分
    Xr_train, Xr_test = X_radiomics[train_idx], X_radiomics[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    # ===== Step 1: 标准化 (仅训练折) =====
    scaler_r = StandardScaler()
    Xr_train_s = scaler_r.fit_transform(Xr_train)
    Xr_test_s = scaler_r.transform(Xr_test)

    lasso = LogisticRegressionCV(
        Cs=np.logspace(-3, 1, 20),
        penalty="l1",
        solver="liblinear",
        cv=inner_cv_fs,
        scoring="roc_auc",
        max_iter=5000,
        class_weight="balanced",
        random_state=42,
        refit=True
    )

    lasso.fit(Xr_train_s, y_train)
    coef = lasso.coef_[0]
    selected_idx = np.where(coef != 0)[0]

    if len(selected_idx) == 0:
        print("⚠️ No features selected, skipping fold")
        fold_id += 1
        continue

    selected_features = [radiomics_feature_names[i] for i in selected_idx]
    feature_stability["Radiomics_Only"].update(
        [radiomics_feature_names[i] for i in selected_idx]
    )
    # ===== Clinical =====
    Xc_train, Xc_test = X_clinical[train_idx], X_clinical[test_idx]

    scaler_c = StandardScaler()
    Xc_train_s = scaler_c.fit_transform(Xc_train)
    Xc_test_s = scaler_c.transform(Xc_test)

    # ===== Radiomics after LASSO =====
    Xr_train_sel = Xr_train_s[:, selected_idx]
    Xr_test_sel = Xr_test_s[:, selected_idx]

    pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    # ===== Step 3: 构建模型参数空间（inner_cv_tune）=====
    models = {
        "LR": {
            "model": LogisticRegression(max_iter=3000, class_weight='balanced', solver="liblinear"),
            "params": {"C": [0.001,0.01,0.05,0.1,0.5,1,2,5,10]}
        },

        "RF": {
            "model": RandomForestClassifier(
                class_weight='balanced',
                random_state=42
            ),
            "params": {
                "n_estimators": [200, 300],
                "max_depth": [3, 5],
                "min_samples_leaf": [1, 2, 4],
                "max_features": ["sqrt", 0.5]
            }
        },

        "XGB": {
            "model": XGBClassifier(
                eval_metric='logloss',
                random_state=42,
                scale_pos_weight=pos_weight
            ),
            "params": {
                "n_estimators": [100, 200],
                "max_depth": [2, 3],
                "learning_rate": [0.01, 0.05],
                "subsample": [0.8, 1.0],
                "colsample_bytree": [0.8, 1.0]
            }
        }
    }

    # ===== Step 4: 内层调参 + 记录 inner AUC =====
    for model_name, cfg in models.items():
        grid = GridSearchCV(
            cfg["model"],
            cfg["params"],
            cv=inner_cv_tune,
            scoring="roc_auc",
            n_jobs=-1
        )

        for set_name, Xtr, Xte in [
            ("Radiomics_Only",
             Xr_train_sel,
             Xr_test_sel),

            ("Radiomics_+_Clinical",
             np.hstack([Xr_train_sel, Xc_train_s]),
             np.hstack([Xr_test_sel, Xc_test_s]))
        ]:
            grid.fit(Xtr, y_train)
            best_model = grid.best_estimator_
            inner_auc = grid.best_score_  # ⭐ inner loop performance
            ###########
            # =========================
            # SHAP（只对 outer test）
            # =========================
            X_shap = Xte  # 注意：已经是 outer test + 正确标准化

            if model_name in ["RF", "XGB"]:
                explainer = shap.TreeExplainer(best_model)
                shap_values = explainer.shap_values(X_shap)

                if isinstance(shap_values, list):
                    shap_values = shap_values[1]
                elif shap_values.ndim == 3:
                    shap_values = shap_values[:, :, 1]

            else:  # Logistic Regression
                explainer = shap.LinearExplainer(
                    best_model,
                    Xtr,  # background = outer train
                    feature_perturbation="interventional"
                )
                shap_values = explainer.shap_values(X_shap)

            # 存 outer fold 的 SHAP
            # shap_storage[set_name][model_name].append(shap_values)
            # =========================
            # 保存 SHAP（带特征名）
            # =========================
            if set_name == "Radiomics_Only":
                feat_names = selected_features
            else:
                feat_names = selected_features + clinical_feature_names

            # 统一 SHAP 维度（RF / XGB 二分类）
            if isinstance(shap_values, list):
                shap_values = shap_values[1]

            shap_storage[set_name][model_name].append({
                "shap": shap_values,  # shape: (n_samples, n_features_this_fold)
                "features": feat_names  # 对应特征名
            })

            #############
            y_prob = best_model.predict_proba(Xte)[:, 1]
            y_pred = best_model.predict(Xte)
            test_auc = roc_auc_score(y_test, y_prob)
            # ⭐ 存 ROC 用的数据（外层测试集）
            roc_storage[set_name][model_name].append((y_test.copy(), y_prob.copy()))
            results.append({
                "Feature_Set": set_name,
                "Outer_Fold": fold_id,
                "Model": model_name,
                # "Selected_Feature_Count": len(selected_idx),
                "Selected_Features": ", ".join(selected_features),
                "Selected_Radiomics_Count": len(selected_idx),
                "Clinical_Count": len(clinical_feature_names) if set_name == "Radiomics_+_Clinical" else 0,
                "Best_Params": str(grid.best_params_),
                "Inner_AUC": inner_auc,
                "Test_AUC": test_auc,
                "Accuracy": accuracy_score(y_test, y_pred),
                "Sensitivity": recall_score(y_test, y_pred),
                "Specificity": recall_score(y_test, y_pred, pos_label=0),
                "Best_C_LASSO": lasso.C_[0],
                "Overfit_Flag": "Yes" if (inner_auc - test_auc) > 0.1 else "No",
                "Generalization_Gap": inner_auc - test_auc

            })

    fold_id += 1
# =========================
# 保存详细结果
# =========================
results_df = pd.DataFrame(results)
out_file = f"E:/US-MRI/{folder_name}/random18/Ultimate_NestedCV_Results.xlsx"
results_df.to_excel(out_file, index=False)
print(f"\n✅ Detailed results saved: {out_file}")
from sklearn.utils import resample

# =========================
# 论文级汇总表（均值 ± 95% CI）
# =========================
def bootstrap_ci(y_vals, n_bootstrap=1000, ci_level=0.95):
    rng = np.random.RandomState(42)
    boot_means = []
    y_vals = np.array(y_vals)
    for _ in range(n_bootstrap):
        sample = resample(y_vals, replace=True, n_samples=len(y_vals), random_state=rng)
        boot_means.append(sample.mean())
    lower = np.percentile(boot_means, (1 - ci_level)/2 * 100)
    upper = np.percentile(boot_means, (1 + ci_level)/2 * 100)
    mean_val = y_vals.mean()
    return mean_val, lower, upper

summary_rows = []
metrics = ["Inner_AUC", "Test_AUC", "Accuracy", "Sensitivity", "Specificity"]

for set_name in results_df["Feature_Set"].unique():
    for model_name in results_df["Model"].unique():
        sub = results_df[
            (results_df["Feature_Set"] == set_name) &
            (results_df["Model"] == model_name)
        ]
        if len(sub) == 0:
            continue

        row = {"Feature_Set": set_name, "Model": model_name}
        for m in metrics:
            mean_val, lower, upper = bootstrap_ci(sub[m].values)
            row[m] = f"{mean_val:.3f} ({lower:.3f}-{upper:.3f})"
        summary_rows.append(row)

summary_df = pd.DataFrame(summary_rows)
summary_file = f"E:/US-MRI/{folder_name}/random18/Ultimate_NestedCV_PaperTable_95CI.xlsx"
summary_df.to_excel(summary_file, index=False)
print(f"✅ Paper table (95% CI) saved: {summary_file}")

stab_df = pd.DataFrame.from_dict(
    feature_stability["Radiomics_Only"],
    orient='index',
    columns=['Selection_Frequency']
).sort_values(by="Selection_Frequency", ascending=False)

stab_file = f"E:/US-MRI/{folder_name}/random18/Ultimate_Feature_Stability_Radiomics_Only.xlsx"
stab_df.to_excel(stab_file)
print(f"✅ Feature stability saved: {stab_file}")

# =========================
# 自动选择 Test AUC 最高的模型
# =========================
best_models = {}

for set_name in results_df["Feature_Set"].unique():
    sub = results_df[results_df["Feature_Set"] == set_name]

    mean_auc_by_model = sub.groupby("Model")["Test_AUC"].mean()
    best_model = mean_auc_by_model.idxmax()

    best_models[set_name] = best_model

    print(f"✅ Best model for {set_name}: {best_model}")
# 保存 best model 的 outer test predictions（用于 AUC 差异检验）
# =========================
import numpy as np
import os

os.makedirs("results", exist_ok=True)

for set_name, best_model_name in best_models.items():

    roc_list = roc_storage[set_name][best_model_name]

    y_true_all = np.concatenate([y for y, _ in roc_list])
    y_prob_all = np.concatenate([p for _, p in roc_list])

    # patient_id（如果你 outer loop 里没存，就先不写）
    np.savez(
        f"E:/US-MRI/{folder_name}/random18/results/{set_name}_{best_model_name}_outer_test_predictions.npz",
        y_true=y_true_all,
        y_prob=y_prob_all
    )

    print(f"✅ Saved outer-test predictions: {set_name} ({best_model_name})")
plt.figure(figsize=(7, 7))
# 设置统一字体大小
font_size = 14
alpha_ci = 0.15  # 阴影透明度调淡
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]  # 可按模型/特征集扩展
# for set_name, model_name in best_models.items():
#     plot_mean_roc_bootstrap(
#         roc_storage[set_name][model_name],
#         label=f"{set_name.replace('_', ' ')} ({model_name})"
#     )
# 绘制 ROC
# 绘制 ROC
for i, (set_name, model_name) in enumerate(best_models.items()):
    color = colors[i % len(colors)]
    roc_data = roc_storage[set_name][model_name]
    plot_mean_roc_bootstrap_custom(roc_data, f"{set_name.replace('_', ' ')} ({model_name})", color, alpha_ci)
plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.xlabel("False Positive Rate", fontsize=font_size)
plt.ylabel("True Positive Rate", fontsize=font_size)
plt.title("ROC Curves of Best-performing Radiomics Models", fontsize=font_size+2)
plt.xticks(fontsize=font_size-2)
plt.yticks(fontsize=font_size-2)
# 添加图例，并设置字号
plt.legend(fontsize=10)  # 12可以根据需要调整
plt.legend(loc="lower right", fontsize=font_size)
plt.tight_layout()
plt.savefig("E:/US-MRI/fe/random18/radiomics_ROC.png", dpi=300)

# =====================================================
# SHAP 汇总（仅解释 best model）
# =====================================================
from collections import defaultdict

for set_name in feature_sets:

    best_model_name = best_models[set_name]
    shap_items = shap_storage[set_name][best_model_name]

    shap_dict = defaultdict(list)

    for item in shap_items:
        shap_vals = item["shap"]      # (n_samples, n_features)
        feats = item["features"]

        for i, f in enumerate(feats):
            shap_dict[f].append(np.abs(shap_vals[:, i]))

    shap_summary = {
        f: np.mean(np.concatenate(v))
        for f, v in shap_dict.items()
    }

    shap_df = (
        pd.DataFrame(
            shap_summary.items(),
            columns=["Feature", "Mean |SHAP value|"]
        )
        .sort_values(by="Mean |SHAP value|", ascending=False)
    )

    # 保存
    shap_df.to_excel(
        f"E:/US-MRI/{folder_name}/random18/SHAP_{set_name}_{best_model_name}.xlsx",
        index=False
    )

    print(f"[SHAP] {set_name} ({best_model_name}) saved.")
###########直接量化特征
import pandas as pd
import numpy as np
import shap
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RepeatedStratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from collections import Counter
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import matplotlib
matplotlib.use('Agg')  # 不使用窗口，直接保存图像

def plot_mean_roc_bootstrap_custom(roc_list, label, color, alpha_ci=0.15, ci_level=0.95):
    """
    roc_list: list of (y_true, y_prob), each from ONE outer CV fold
    """

    mean_fpr = np.linspace(0, 1, 100)
    tprs = []
    aucs = []

    # ===== 1. 每个 CV fold 计算一条 ROC =====
    for y_true, y_prob in roc_list:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        fold_auc = auc(fpr, tpr)
        aucs.append(fold_auc)

        tpr_interp = np.interp(mean_fpr, fpr, tpr)
        tpr_interp[0] = 0.0
        tprs.append(tpr_interp)

    tprs = np.array(tprs)
    aucs = np.array(aucs)

    # ===== 2. mean ROC =====
    mean_tpr = tprs.mean(axis=0)
    mean_tpr[-1] = 1.0

    # ===== 3. 95% CI of ROC (CV-level percentile) =====
    lower_tpr = np.percentile(
        tprs, (1 - ci_level) / 2 * 100, axis=0
    )
    upper_tpr = np.percentile(
        tprs, (1 + ci_level) / 2 * 100, axis=0
    )

    # ===== 4. AUC = mean(Test_AUC) + 95% CI =====
    mean_auc = aucs.mean()
    lower_auc = np.percentile(
        aucs, (1 - ci_level) / 2 * 100
    )
    upper_auc = np.percentile(
        aucs, (1 + ci_level) / 2 * 100
    )

    # ===== 5. Plot =====
    plt.plot(
        mean_fpr,
        mean_tpr,
        lw=2,
        color=color,
        label=f"{label} (AUC = {mean_auc:.3f})"
    )
    plt.fill_between(
        mean_fpr,
        lower_tpr,
        upper_tpr,
        color=color,
        alpha=alpha_ci
    )

# =========================
# 参数配置
# =========================
target_var = "Diabetes_Status"

quantitative_features = [
    "IMAT_to_Muscle_Ratio_pct",
    "Subcutaneous_to_Muscle_Ratio_pct",
    "IMAT_Fraction_pct",

    # Tissue composition
    "Muscle_pct_Tissue",
    "Fat_pct_Tissue",
    "IMAT_pct_Tissue",
    "Subcutaneous_pct_Tissue",
    "Fat_to_Muscle_Ratio",

    # SMFF metrics
    "SMFF_mean",
    "SMFF_std",
    "SMFF_median",
    "SMFF_25perc",
    "SMFF_75perc",
    "SMFF_wf"
]
clinical_feature_names = ["BMI", "Age", "Sex"]

feature_sets = {
    "Quantitative_Only": quantitative_features,
    "Quantitative_+_Clinical": quantitative_features + clinical_feature_names
}

folder_name = "fe"
data_file = f"E:/US-MRI/{folder_name}/direc_quan.xlsx"
Status_file = f"E:/US-MRI/{folder_name}/1.xlsx"
id_col = "Patient_ID"

# =========================
# 读取数据
# =========================
df_feat = pd.read_excel(data_file, engine="openpyxl")
df_split = pd.read_excel(Status_file, engine="openpyxl")

df = pd.merge(
    df_split[[id_col, target_var] + clinical_feature_names],
    df_feat[[id_col] + quantitative_features],
    on=id_col,
    how="inner"
)

# 安全校验
missing_cols = [col for col in [target_var] + clinical_feature_names + quantitative_features if col not in df.columns]
if missing_cols:
    raise ValueError(f"❌ Missing columns after merge: {missing_cols}")

print("✅ Merged dataset shape:", df.shape)

# =========================
# 0️⃣ 单变量 Logistic Regression
# =========================
print("\n===== 单变量 Logistic Regression =====")
univariate_results = []

for feat in quantitative_features:
    X = df[[feat]].values
    y = df[target_var].values

    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    # sklearn LR
    lr = LogisticRegression(class_weight="balanced", solver="liblinear")
    lr.fit(X_s, y)
    y_prob = lr.predict_proba(X_s)[:, 1]
    y_pred = lr.predict(X_s)

    auc_val = roc_auc_score(y, y_prob)
    acc = accuracy_score(y, y_pred)
    sen = recall_score(y, y_pred)
    spe = recall_score(y, y_pred, pos_label=0)

    # statsmodels Logit OR / CI / p
    X_const = sm.add_constant(X_s)
    logit = sm.Logit(y, X_const).fit(disp=False)

    coef = logit.params[1]  # 特征系数
    OR = np.exp(coef)

    conf = logit.conf_int()
    if isinstance(conf, np.ndarray):
        CI_low, CI_high = np.exp(conf[1, 0]), np.exp(conf[1, 1])
    else:
        CI_low, CI_high = np.exp(conf.iloc[1, 0]), np.exp(conf.iloc[1, 1])

    p = logit.pvalues[1]

    univariate_results.append({
        "Feature": feat,
        "AUC": auc_val,
        "Accuracy": acc,
        "Sensitivity": sen,
        "Specificity": spe,
        "OR": OR,
        "CI_low": CI_low,
        "CI_high": CI_high,
        "p_value": p
    })

# 保存单变量结果
univariate_df = pd.DataFrame(univariate_results)
univariate_file = f"E:/US-MRI/{folder_name}/direc_Univariate_LR_Results.xlsx"
univariate_df.to_excel(univariate_file, index=False)
print(f"✅ Univariate LR results saved: {univariate_file}")

# =========================
# Nested CV 设置
# =========================
outer_cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=42)
inner_cv_fs = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=1)
inner_cv_tune = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=7)

# 结果容器
results = []
feature_stability = {k: Counter() for k in feature_sets.keys()}
# =========================
# ROC 数据容器（外层测试集）
# =========================
roc_storage = {
    "Quantitative_Only": {"LR": [], "RF": [], "XGB": []},
    "Quantitative_+_Clinical": {"LR": [], "RF": [], "XGB": []}
}
# 每个元素是 (y_test, y_prob)

# =========================
# 模型参数空间
# =========================
models_config = {
    "LR": {
        "model": LogisticRegression(max_iter=3000, class_weight='balanced', solver="liblinear"),
        "params": {"C": [0.01, 0.1, 1, 5]}
    },
    "RF": {
        "model": RandomForestClassifier(class_weight='balanced', random_state=42),
        "params": {"n_estimators": [200, 300], "max_depth": [3, 5]}
    },
    "XGB": {
        "model": None,  # 会在循环中生成
        "params": {"n_estimators": [200, 300], "max_depth": [3, 4], "learning_rate": [0.01, 0.05]}
    }
}

# =========================
# Nested CV 主循环
# =========================
# for set_name, features in feature_sets.items():
#     print(f"\n==== Feature Set: {set_name} ====")
#     X = df[features].values
#     y = df[target_var].values
X_direct = df[quantitative_features].values
X_clinical = df[clinical_feature_names].values
y = df[target_var].values
fold_id = 1

shap_storage = {
    "Quantitative_Only": {"LR": [], "RF": [], "XGB": []},
    "Quantitative_+_Clinical": {"LR": [], "RF": [], "XGB": []}
}

for train_idx, test_idx in outer_cv.split(X_direct, y):
    print(f"Outer Fold {fold_id}")

    # X_train, X_test = X[train_idx], X[test_idx]
    # y_train, y_test = y[train_idx], y[test_idx]
    #
    # # 标准化
    # scaler = StandardScaler()
    # X_train_s = scaler.fit_transform(X_train)
    # X_test_s = scaler.transform(X_test)
    # 外层划分
    Xr_train, Xr_test = X_direct[train_idx], X_direct[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    # ===== Step 1: 标准化 (仅训练折) =====
    scaler_r = StandardScaler()
    Xr_train_s = scaler_r.fit_transform(Xr_train)
    Xr_test_s = scaler_r.transform(Xr_test)
    # LASSO Logistic 特征选择
    lasso = LogisticRegressionCV(
        Cs=np.logspace(-3, 1, 20),
        penalty="l1",
        solver="liblinear",
        cv=inner_cv_fs,
        scoring="roc_auc",
        max_iter=5000,
        class_weight="balanced",
        random_state=42,
        refit=True
    )
    # lasso.fit(X_train_s, y_train)
    lasso.fit(Xr_train_s, y_train)
    coef = lasso.coef_[0]
    selected_idx = np.where(coef != 0)[0]
    if len(selected_idx) == 0:
        print("⚠️ No features selected, skipping fold")
        fold_id += 1
        continue

    # selected_features = [features[i] for i in selected_idx]
    # feature_stability[set_name].update(selected_features)
    #
    # X_train_sel = X_train_s[:, selected_idx]
    # X_test_sel = X_test_s[:, selected_idx]
    selected_features = [quantitative_features[i] for i in selected_idx]
    feature_stability["Quantitative_Only"].update(
        [quantitative_features[i] for i in selected_idx]
    )
    # ===== Clinical =====
    Xc_train, Xc_test = X_clinical[train_idx], X_clinical[test_idx]

    scaler_c = StandardScaler()
    Xc_train_s = scaler_c.fit_transform(Xc_train)
    Xc_test_s = scaler_c.transform(Xc_test)

    # ===== Radiomics after LASSO =====
    Xr_train_sel = Xr_train_s[:, selected_idx]
    Xr_test_sel = Xr_test_s[:, selected_idx]

    pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    # 训练 LR / RF / XGB
    for model_name in ["LR", "RF", "XGB"]:
        if model_name == "XGB":
            cfg_model = XGBClassifier(eval_metric='logloss', random_state=42, scale_pos_weight=pos_weight)
        else:
            cfg_model = models_config[model_name]["model"]
        grid = GridSearchCV(
            cfg_model,
            models_config[model_name]["params"],
            cv=inner_cv_tune,
            scoring="roc_auc",
            n_jobs=-1
        )
        for set_name, Xtr, Xte in [
            ("Quantitative_Only",
             Xr_train_sel,
             Xr_test_sel),

            ("Quantitative_+_Clinical",
             np.hstack([Xr_train_sel, Xc_train_s]),
             np.hstack([Xr_test_sel, Xc_test_s]))
        ]:
            grid.fit(Xtr, y_train)
            best_model = grid.best_estimator_
            inner_auc = grid.best_score_  # ⭐ inner loop performance
            ###########
            # =========================
            # SHAP（只对 outer test）
            # =========================
            X_shap = Xte  # 注意：已经是 outer test + 正确标准化

            if model_name in ["RF", "XGB"]:
                explainer = shap.TreeExplainer(best_model)
                shap_values = explainer.shap_values(X_shap)

                if isinstance(shap_values, list):
                    shap_values = shap_values[1]
                elif shap_values.ndim == 3:
                    shap_values = shap_values[:, :, 1]

            else:  # Logistic Regression
                explainer = shap.LinearExplainer(
                    best_model,
                    Xtr,  # background = outer train
                    feature_perturbation="interventional"
                )
                shap_values = explainer.shap_values(X_shap)

            # 存 outer fold 的 SHAP
            # shap_storage[set_name][model_name].append(shap_values)
            # =========================
            # 保存 SHAP（带特征名）
            # =========================
            if set_name == "Quantitative_Only":
                feat_names = selected_features
            else:
                feat_names = selected_features + clinical_feature_names

            # 统一 SHAP 维度（RF / XGB 二分类）
            if isinstance(shap_values, list):
                shap_values = shap_values[1]

            shap_storage[set_name][model_name].append({
                "shap": shap_values,  # shape: (n_samples, n_features_this_fold)
                "features": feat_names  # 对应特征名
            })
            ############
            y_prob = best_model.predict_proba(Xte)[:, 1]
            y_pred = best_model.predict(Xte)
            test_auc = roc_auc_score(y_test, y_prob)
            # ⭐ 存 ROC 用的数据（外层测试集）
            roc_storage[set_name][model_name].append((y_test.copy(), y_prob.copy()))
            results.append({
                "Feature_Set": set_name,
                "Outer_Fold": fold_id,
                "Model": model_name,
                "Selected_Features": ", ".join(selected_features),
                "Selected_Radiomics_Count": len(selected_idx),
                "Clinical_Count": len(clinical_feature_names) if set_name == "Quantitative_+_Clinical" else 0,
                "Best_Params": str(grid.best_params_),
                "Inner_AUC": inner_auc,
                "Test_AUC": test_auc,
                "Accuracy": accuracy_score(y_test, y_pred),
                "Sensitivity": recall_score(y_test, y_pred),
                "Specificity": recall_score(y_test, y_pred, pos_label=0),
                "Best_C_LASSO": lasso.C_[0],
                "Overfit_Flag": "Yes" if (inner_auc - test_auc) > 0.1 else "No",
                "Generalization_Gap": inner_auc - test_auc

            })
    fold_id += 1

# =========================
# 保存 Nested CV 详细结果
# =========================
results_df = pd.DataFrame(results)
out_file = f"E:/US-MRI/{folder_name}/direc_Ultimate_NestedCV_Results.xlsx"
results_df.to_excel(out_file, index=False)
print(f"\n✅ Detailed Nested CV results saved: {out_file}")
from sklearn.utils import resample

# =========================
# 论文级汇总表（均值 ± 95% CI）
# =========================
def bootstrap_ci(y_vals, n_bootstrap=1000, ci_level=0.95):
    rng = np.random.RandomState(42)
    boot_means = []
    y_vals = np.array(y_vals)
    for _ in range(n_bootstrap):
        sample = resample(y_vals, replace=True, n_samples=len(y_vals), random_state=rng)
        boot_means.append(sample.mean())
    lower = np.percentile(boot_means, (1 - ci_level)/2 * 100)
    upper = np.percentile(boot_means, (1 + ci_level)/2 * 100)
    mean_val = y_vals.mean()
    return mean_val, lower, upper

summary_rows = []
metrics = ["Inner_AUC", "Test_AUC", "Accuracy", "Sensitivity", "Specificity"]

for set_name in results_df["Feature_Set"].unique():
    for model_name in results_df["Model"].unique():
        sub = results_df[
            (results_df["Feature_Set"] == set_name) &
            (results_df["Model"] == model_name)
        ]
        if len(sub) == 0:
            continue

        row = {"Feature_Set": set_name, "Model": model_name}
        for m in metrics:
            mean_val, lower, upper = bootstrap_ci(sub[m].values)
            row[m] = f"{mean_val:.3f} ({lower:.3f}-{upper:.3f})"
        summary_rows.append(row)

summary_df = pd.DataFrame(summary_rows)
summary_file = f"E:/US-MRI/{folder_name}/direc_Ultimate_NestedCV_PaperTable_95CI.xlsx"
summary_df.to_excel(summary_file, index=False)
print(f"✅ Paper table (95% CI) saved: {summary_file}")

stab_df = pd.DataFrame.from_dict(
    feature_stability["Quantitative_Only"],
    orient='index',
    columns=['Selection_Frequency']
).sort_values(by="Selection_Frequency", ascending=False)

stab_file = f"E:/US-MRI/{folder_name}/Ultimate_Feature_Stability_Quantitative_Only.xlsx"
stab_df.to_excel(stab_file)
print(f"✅ Feature stability saved: {stab_file}")
# =========================
# 自动选择 Test AUC 最高的模型
# =========================
best_models = {}

for set_name in results_df["Feature_Set"].unique():
    sub = results_df[results_df["Feature_Set"] == set_name]

    mean_auc_by_model = sub.groupby("Model")["Test_AUC"].mean()
    best_model = mean_auc_by_model.idxmax()

    best_models[set_name] = best_model

    print(f"✅ Best model for {set_name}: {best_model}")
# =========================
# 保存 best model 的 outer test predictions（用于 AUC 差异检验）
# =========================
import numpy as np
import os

os.makedirs("results", exist_ok=True)

for set_name, best_model_name in best_models.items():

    roc_list = roc_storage[set_name][best_model_name]

    y_true_all = np.concatenate([y for y, _ in roc_list])
    y_prob_all = np.concatenate([p for _, p in roc_list])

    # patient_id（如果你 outer loop 里没存，就先不写）
    np.savez(
        f"E:/US-MRI/{folder_name}/results/{set_name}_{best_model_name}_outer_test_predictions.npz",
        y_true=y_true_all,
        y_prob=y_prob_all
    )

    print(f"✅ Saved outer-test predictions: {set_name} ({best_model_name})")



plt.figure(figsize=(7, 7))
# 设置统一字体大小
font_size = 14
alpha_ci = 0.15  # 阴影透明度调淡
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]  # 可按模型/特征集扩展
# 绘制 ROC
for i, (set_name, model_name) in enumerate(best_models.items()):
    color = colors[i % len(colors)]
    roc_data = roc_storage[set_name][model_name]
    plot_mean_roc_bootstrap_custom(roc_data, f"{set_name.replace('_', ' ')} ({model_name})", color, alpha_ci)
plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.xlabel("False Positive Rate", fontsize=font_size)
plt.ylabel("True Positive Rate", fontsize=font_size)
plt.title("ROC Curves of Best-performing Quantitative Models", fontsize=font_size+2)
plt.xticks(fontsize=font_size-2)
plt.yticks(fontsize=font_size-2)
# 添加图例，并设置字号
plt.legend(fontsize=10)  # 12可以根据需要调整
plt.legend(loc="lower right", fontsize=font_size)
plt.tight_layout()
plt.savefig("E:/US-MRI/fe/direct_ROC.png", dpi=300)
# =====================================================
# SHAP 汇总（仅解释 best model）
# =====================================================
from collections import defaultdict

for set_name in feature_sets:

    best_model_name = best_models[set_name]
    shap_items = shap_storage[set_name][best_model_name]

    shap_dict = defaultdict(list)

    for item in shap_items:
        shap_vals = item["shap"]      # (n_samples, n_features)
        feats = item["features"]

        for i, f in enumerate(feats):
            shap_dict[f].append(np.abs(shap_vals[:, i]))

    shap_summary = {
        f: np.mean(np.concatenate(v))
        for f, v in shap_dict.items()
    }

    shap_df = (
        pd.DataFrame(
            shap_summary.items(),
            columns=["Feature", "Mean |SHAP value|"]
        )
        .sort_values(by="Mean |SHAP value|", ascending=False)
    )

    # 保存
    shap_df.to_excel(
        f"E:/US-MRI/{folder_name}/SHAP_{set_name}_{best_model_name}.xlsx",
        index=False
    )

    print(f"[SHAP] {set_name} ({best_model_name}) saved.")

# #################全特征的
# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import RepeatedStratifiedKFold, GridSearchCV
# from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
# from sklearn.metrics import roc_auc_score, accuracy_score, recall_score
# from sklearn.ensemble import RandomForestClassifier
# from xgboost import XGBClassifier
# from collections import Counter
# import matplotlib.pyplot as plt
# from sklearn.metrics import roc_curve, auc
# import matplotlib
# import shap
# matplotlib.use('Agg')  # 不使用窗口，直接保存图像
#
# def plot_mean_roc_bootstrap_custom(roc_list, label, color, alpha_ci=0.15, ci_level=0.95):
#     """
#     roc_list: list of (y_true, y_prob), each from ONE outer CV fold
#     """
#
#     mean_fpr = np.linspace(0, 1, 100)
#     tprs = []
#     aucs = []
#
#     # ===== 1. 每个 CV fold 计算一条 ROC =====
#     for y_true, y_prob in roc_list:
#         fpr, tpr, _ = roc_curve(y_true, y_prob)
#         fold_auc = auc(fpr, tpr)
#         aucs.append(fold_auc)
#
#         tpr_interp = np.interp(mean_fpr, fpr, tpr)
#         tpr_interp[0] = 0.0
#         tprs.append(tpr_interp)
#
#     tprs = np.array(tprs)
#     aucs = np.array(aucs)
#
#     # ===== 2. mean ROC =====
#     mean_tpr = tprs.mean(axis=0)
#     mean_tpr[-1] = 1.0
#
#     # ===== 3. 95% CI of ROC (CV-level percentile) =====
#     lower_tpr = np.percentile(
#         tprs, (1 - ci_level) / 2 * 100, axis=0
#     )
#     upper_tpr = np.percentile(
#         tprs, (1 + ci_level) / 2 * 100, axis=0
#     )
#
#     # ===== 4. AUC = mean(Test_AUC) + 95% CI =====
#     mean_auc = aucs.mean()
#     lower_auc = np.percentile(
#         aucs, (1 - ci_level) / 2 * 100
#     )
#     upper_auc = np.percentile(
#         aucs, (1 + ci_level) / 2 * 100
#     )
#
#     # ===== 5. Plot =====
#     plt.plot(
#         mean_fpr,
#         mean_tpr,
#         lw=2,
#         color=color,
#         label=f"{label} (AUC = {mean_auc:.3f})"
#     )
#     plt.fill_between(
#         mean_fpr,
#         lower_tpr,
#         upper_tpr,
#         color=color,
#         alpha=alpha_ci
#     )
#
# # =========================
# # 参数配置
# # =========================
# target_var = "Diabetes_Status"
# radiomics_feature_names = [

# #                         "Muscle_Fat_original_firstorder_InterquartileRange",
# #                         "Muscle_Fat_original_firstorder_Kurtosis",
# #                         "Muscle_Fat_original_firstorder_Mean",
# #                         "Muscle_Fat_original_firstorder_Median",
# #                         "Muscle_Fat_original_firstorder_Skewness",
# #                         "Muscle_Fat_original_firstorder_Variance",
# #                         "IMAT_Fat_original_firstorder_InterquartileRange",
# #                         "IMAT_Fat_original_firstorder_Kurtosis",
# #                         "IMAT_Fat_original_firstorder_Mean",
# #                         "IMAT_Fat_original_firstorder_Median",
# #                         "IMAT_Fat_original_firstorder_Skewness",
# #                         "IMAT_Fat_original_firstorder_Variance",
# #                         "Subcutaneous_Fat_Fat_original_firstorder_InterquartileRange",
# #                         "Subcutaneous_Fat_Fat_original_firstorder_Kurtosis",
# #                         "Subcutaneous_Fat_Fat_original_firstorder_Mean",
# #                         "Subcutaneous_Fat_Fat_original_firstorder_Median",
# #                         "Subcutaneous_Fat_Fat_original_firstorder_Skewness",
# #                         "Subcutaneous_Fat_Fat_original_firstorder_Variance",
# #
# #                         "Muscle_original_shape_Flatness",
# #                         "Muscle_original_shape_Sphericity",
# #                         "IMAT_original_shape_Flatness",
# #                         "IMAT_original_shape_Sphericity",
# #                         "Subcutaneous_Fat_original_shape_Flatness",
# #                         "Subcutaneous_Fat_original_shape_Sphericity",
# #
# #                         "Muscle_Fat_original_glcm_Imc1",
# #                         "Muscle_Water_original_glcm_SumEntropy",
# #                         "IMAT_Fat_original_glcm_ClusterShade",
# #                         "Subcutaneous_Fat_Fat_original_glcm_ClusterProminence",
# #                         "IMAT_Water_original_glcm_ClusterTendency",
# #                         "IMAT_Water_original_glcm_ClusterShade",
# #                         "IMAT_Fat_original_glcm_MaximumProbability",
# #                         "Muscle_Fat_original_glcm_MCC",
# #                         "Muscle_Fat_original_glcm_InverseVariance",
# #                         "IMAT_Water_original_glcm_MaximumProbability",
# #                         "IMAT_to_Muscle_Ratio_pct",
# #                         "Subcutaneous_to_Muscle_Ratio_pct",
# #                         "IMAT_Fraction_pct",
# #
# #                         # Tissue composition
# #                         "Muscle_pct_Tissue",
# #                         "Fat_pct_Tissue",
# #                         "IMAT_pct_Tissue",
# #                         "Subcutaneous_pct_Tissue",
# #                         "Fat_to_Muscle_Ratio",
# #
# #                         # SMFF metrics
# #                         "SMFF_mean",
# #                         "SMFF_std",
# #                         "SMFF_median",
# #                         "SMFF_25perc",
# #                         "SMFF_75perc",
# #                         "SMFF_wf"
# ]

# ]
#
# clinical_feature_names = ["BMI", "Age", "Sex"]
#
# feature_sets = {
#     "All-feature_Only": radiomics_feature_names,
#     "All-feature_+_Clinical": radiomics_feature_names + clinical_feature_names
# }
#
# folder_name = "fe"
# data_file = f"E:/US-MRI/{folder_name}/all.xlsx"
# # 标签文件（含 Diabetes_Status）
# Status_file = f"E:/US-MRI/{folder_name}/1.xlsx"
# id_col = "Patient_ID"
# # =========================
# # 读取数据
# # =========================
# df_feat = pd.read_excel(data_file, engine="openpyxl")
# df_split = pd.read_excel(Status_file, engine="openpyxl")
#
# df = pd.merge(
#     df_split[[id_col, target_var] + clinical_feature_names],
#     df_feat[[id_col] + radiomics_feature_names],
#     on=id_col,
#     how="inner"
# )
#
# # 安全校验（防止论文审稿人质疑）
# missing_cols = [col for col in [target_var] + clinical_feature_names + radiomics_feature_names if col not in df.columns]
# if missing_cols:
#     raise ValueError(f"❌ Missing columns after merge: {missing_cols}")
#
# print("✅ Merged dataset shape:", df.shape)
# # =========================
# # 全训练集特征相关性分析（方法学检查）
# # =========================
# from itertools import combinations
#
# corr_df = df[radiomics_feature_names]
#
# # Spearman 相关性（适合放射组学）
# corr_matrix = corr_df.corr(method="spearman")
#
# # 取绝对值
# corr_abs = corr_matrix.abs()
#
# # 找高相关特征对（|ρ| > 0.8）
# high_corr_pairs = []
#
# for i, j in combinations(corr_abs.columns, 2):
#     if corr_abs.loc[i, j] > 0.8:
#         high_corr_pairs.append({
#             "Feature_1": i,
#             "Feature_2": j,
#             "Spearman_r": corr_matrix.loc[i, j]
#         })
#
# high_corr_df = pd.DataFrame(high_corr_pairs)
#
# # 保存（建议作为 Supplementary Table）
# out_corr = f"E:/US-MRI/{folder_name}/all_Spearman_Correlation.xlsx"
# high_corr_df.to_excel(out_corr, index=False)
#
# print(f"✅ Correlation analysis saved: {out_corr}")
#
# if len(high_corr_df) == 0:
#     print("✅ No feature pairs with |Spearman r| > 0.8")
# else:
#     print("⚠️ High correlation pairs detected:")
#     print(high_corr_df)
#
# # =========================
# # 严格 Nested CV 设置
# # =========================
# outer_cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=42)
# inner_cv_fs = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=1)
# inner_cv_tune = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=7)
#
# # =========================
# # 结果容器
# # =========================
# results = []
#
# # =========================
# # ROC 容器（用于论文画 ROC）
# # =========================
# roc_storage = {
#     "All-feature_Only": {"LR": [], "RF": [], "XGB": []},
#     "All-feature_+_Clinical": {"LR": [], "RF": [], "XGB": []}
# }
# # 每个元素存 (y_test, y_prob)
#
# # =========================
# # 主循环
# # =========================
# # for set_name, features in feature_sets.items():
# #     print(f"\n==== Feature Set: {set_name} ====")
# #
# #     X = df[features].values
# #     y = df[target_var].values
# X_radiomics = df[radiomics_feature_names].values
# X_clinical = df[clinical_feature_names].values
# y = df[target_var].values
#
# fold_id = 1
# shap_storage = {
#     "All-feature_Only": {"LR": [], "RF": [], "XGB": []},
#     "All-feature_+_Clinical": {"LR": [], "RF": [], "XGB": []}
# }
#
# for train_idx, test_idx in outer_cv.split(X_radiomics, y):
#     print(f"Outer Fold {fold_id}")
#
#     # 外层划分
#     # X_train, X_test = X[train_idx], X[test_idx]
#     # y_train, y_test = y[train_idx], y[test_idx]
#     #
#     # # ===== Step 1: 标准化 (仅训练折) =====
#     # scaler = StandardScaler()
#     # X_train_s = scaler.fit_transform(X_train)
#     # X_test_s = scaler.transform(X_test)
#     # 外层划分
#     Xr_train, Xr_test = X_radiomics[train_idx], X_radiomics[test_idx]
#     y_train, y_test = y[train_idx], y[test_idx]
#
#     # ===== Step 1: 标准化 (仅训练折) =====
#     scaler_r = StandardScaler()
#     Xr_train_s = scaler_r.fit_transform(Xr_train)
#     Xr_test_s = scaler_r.transform(Xr_test)
#     Xr_train_sel = Xr_train_s
#     Xr_test_sel = Xr_test_s
#     selected_features = radiomics_feature_names
#
#     # ===== Clinical =====
#     Xc_train, Xc_test = X_clinical[train_idx], X_clinical[test_idx]
#
#     scaler_c = StandardScaler()
#     Xc_train_s = scaler_c.fit_transform(Xc_train)
#     Xc_test_s = scaler_c.transform(Xc_test)
#
#
#     pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
#     # ===== Step 3: 构建模型参数空间（inner_cv_tune）=====
#     models = {
#         "LR": {
#             "model": LogisticRegression(max_iter=3000, class_weight='balanced', solver="liblinear"),
#             "params": {"C": [0.01, 0.1, 1, 5]}
#         },
#         "RF": {
#             "model": RandomForestClassifier(class_weight='balanced', random_state=42),
#             "params": {"n_estimators": [200, 300], "max_depth": [3, 5]}
#         },
#         "XGB": {
#             "model": XGBClassifier(
#                 eval_metric='logloss',
#                 random_state=42,
#                 scale_pos_weight=pos_weight
#             ),
#             "params": {"n_estimators": [200, 300], "max_depth": [3, 4], "learning_rate": [0.01, 0.05]}
#             # "model": XGBClassifier(eval_metric='logloss', use_label_encoder=False, random_state=42),
#             # "params": {"n_estimators": [200, 300], "max_depth": [3, 4], "learning_rate": [0.01, 0.05]}
#         }
#     }
#
#     # ===== Step 4: 内层调参 + 记录 inner AUC =====
#     for model_name, cfg in models.items():
#         grid = GridSearchCV(
#             cfg["model"],
#             cfg["params"],
#             cv=inner_cv_tune,
#             scoring="roc_auc",
#             n_jobs=-1
#         )
#
#         # # ===== Step 5: 外层测试集评估 =====
#         for set_name, Xtr, Xte in [
#             ("All-feature_Only",
#              Xr_train_sel,
#              Xr_test_sel),
#
#             ("All-feature_+_Clinical",
#              np.hstack([Xr_train_sel, Xc_train_s]),
#              np.hstack([Xr_test_sel, Xc_test_s]))
#         ]:
#             grid.fit(Xtr, y_train)
#             best_model = grid.best_estimator_
#             inner_auc = grid.best_score_  # ⭐ inner loop performance
#             ###########
#             # =========================
#             # SHAP（只对 outer test）
#             # =========================
#             X_shap = Xte  # 注意：已经是 outer test + 正确标准化
#
#             if model_name in ["RF", "XGB"]:
#                 explainer = shap.TreeExplainer(best_model)
#                 shap_values = explainer.shap_values(X_shap)
#
#                 if isinstance(shap_values, list):
#                     shap_values = shap_values[1]
#                 elif shap_values.ndim == 3:
#                     shap_values = shap_values[:, :, 1]
#
#             else:  # Logistic Regression
#                 explainer = shap.LinearExplainer(
#                     best_model,
#                     Xtr,  # background = outer train
#                     feature_perturbation="interventional"
#                 )
#                 shap_values = explainer.shap_values(X_shap)
#
#             # 存 outer fold 的 SHAP
#             # =========================
#             # 保存 SHAP（带特征名）
#             # =========================
#             if set_name == "All-feature_Only":
#                 feat_names = selected_features
#             else:
#                 feat_names = selected_features + clinical_feature_names
#
#             # 统一 SHAP 维度（RF / XGB 二分类）
#             if isinstance(shap_values, list):
#                 shap_values = shap_values[1]
#
#             shap_storage[set_name][model_name].append({
#                 "shap": shap_values,  # shape: (n_samples, n_features_this_fold)
#                 "features": feat_names  # 对应特征名
#             })
#
#             #############
#             y_prob = best_model.predict_proba(Xte)[:, 1]
#             y_pred = best_model.predict(Xte)
#             test_auc = roc_auc_score(y_test, y_prob)
#             # ⭐ 存 ROC 用的数据（外层测试集）
#             roc_storage[set_name][model_name].append((y_test.copy(), y_prob.copy()))
#             results.append({
#                 "Feature_Set": set_name,
#                 "Outer_Fold": fold_id,
#                 "Model": model_name,
#                 "Clinical_Count": len(clinical_feature_names) if set_name == "All-feature_+_Clinical" else 0,
#                 "Best_Params": str(grid.best_params_),
#                 "Inner_AUC": inner_auc,
#                 "Test_AUC": test_auc,
#                 "Accuracy": accuracy_score(y_test, y_pred),
#                 "Sensitivity": recall_score(y_test, y_pred),
#                 "Specificity": recall_score(y_test, y_pred, pos_label=0),
#                 "Overfit_Flag": "Yes" if (inner_auc - test_auc) > 0.1 else "No",
#                 "Generalization_Gap": inner_auc - test_auc
#
#             })
#
#     fold_id += 1
#
# # =========================
# # 保存详细结果
# # =========================
# results_df = pd.DataFrame(results)
# out_file = f"E:/US-MRI/{folder_name}/All-feature_NestedCV_Results.xlsx"
# results_df.to_excel(out_file, index=False)
# print(f"\n✅ Detailed results saved: {out_file}")
# from sklearn.utils import resample
# # =========================
# # 论文级汇总表（均值 ± 95% CI）
# # =========================
# def bootstrap_ci(y_vals, n_bootstrap=1000, ci_level=0.95):
#     rng = np.random.RandomState(42)
#     boot_means = []
#     y_vals = np.array(y_vals)
#     for _ in range(n_bootstrap):
#         sample = resample(y_vals, replace=True, n_samples=len(y_vals), random_state=rng)
#         boot_means.append(sample.mean())
#     lower = np.percentile(boot_means, (1 - ci_level)/2 * 100)
#     upper = np.percentile(boot_means, (1 + ci_level)/2 * 100)
#     mean_val = y_vals.mean()
#     return mean_val, lower, upper
#
# summary_rows = []
# metrics = ["Inner_AUC", "Test_AUC", "Accuracy", "Sensitivity", "Specificity"]
#
# for set_name in results_df["Feature_Set"].unique():
#     for model_name in results_df["Model"].unique():
#         sub = results_df[
#             (results_df["Feature_Set"] == set_name) &
#             (results_df["Model"] == model_name)
#         ]
#         if len(sub) == 0:
#             continue
#
#         row = {"Feature_Set": set_name, "Model": model_name}
#         for m in metrics:
#             mean_val, lower, upper = bootstrap_ci(sub[m].values)
#             row[m] = f"{mean_val:.3f} ({lower:.3f}-{upper:.3f})"
#         summary_rows.append(row)
#
# summary_df = pd.DataFrame(summary_rows)
# summary_file = f"E:/US-MRI/{folder_name}/All-feature_NestedCV_PaperTable_95CI.xlsx"
# summary_df.to_excel(summary_file, index=False)
# print(f"✅ Paper table (95% CI) saved: {summary_file}")
#
# # =========================
# # 自动选择 Test AUC 最高的模型
# # =========================
# best_models = {}
#
# for set_name in results_df["Feature_Set"].unique():
#     sub = results_df[results_df["Feature_Set"] == set_name]
#
#     mean_auc_by_model = sub.groupby("Model")["Test_AUC"].mean()
#     best_model = mean_auc_by_model.idxmax()
#
#     best_models[set_name] = best_model
#
#     print(f"✅ Best model for {set_name}: {best_model}")
# # =========================
# # 保存 best model 的 outer test predictions（用于 AUC 差异检验）
# # =========================
# import numpy as np
# import os
#
# os.makedirs("results", exist_ok=True)
#
# for set_name, best_model_name in best_models.items():
#
#     roc_list = roc_storage[set_name][best_model_name]
#
#     y_true_all = np.concatenate([y for y, _ in roc_list])
#     y_prob_all = np.concatenate([p for _, p in roc_list])
#
#     # patient_id（如果你 outer loop 里没存，就先不写）
#     np.savez(
#         f"E:/US-MRI/{folder_name}/results/{set_name}_{best_model_name}_outer_test_predictions.npz",
#         y_true=y_true_all,
#         y_prob=y_prob_all
#     )
#
#     print(f"✅ Saved outer-test predictions: {set_name} ({best_model_name})")
# plt.figure(figsize=(7, 7))
# # 设置统一字体大小
# font_size = 14
# alpha_ci = 0.15  # 阴影透明度调淡
# colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]  # 可按模型/特征集扩展
# # 绘制 ROC
# for i, (set_name, model_name) in enumerate(best_models.items()):
#     color = colors[i % len(colors)]
#     roc_data = roc_storage[set_name][model_name]
#     plot_mean_roc_bootstrap_custom(roc_data, f"{set_name.replace('_', ' ')} ({model_name})", color, alpha_ci)
# plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
# plt.xlim(0, 1)
# plt.ylim(0, 1)
# plt.xlabel("False Positive Rate", fontsize=font_size)
# plt.ylabel("True Positive Rate", fontsize=font_size)
# plt.title("ROC Curves of Best-performing All-feature Models", fontsize=font_size+2)
# plt.xticks(fontsize=font_size-2)
# plt.yticks(fontsize=font_size-2)
# # 添加图例，并设置字号
# plt.legend(fontsize=10)  # 12可以根据需要调整
# plt.legend(loc="lower right", fontsize=font_size)
# plt.tight_layout()
# plt.savefig("E:/US-MRI/fe/All-feature_ROC.png", dpi=300)
# # =====================================================
# # SHAP 汇总（仅解释 best model）
# # =====================================================
# from collections import defaultdict
#
# for set_name in feature_sets:
#
#     best_model_name = best_models[set_name]
#     shap_items = shap_storage[set_name][best_model_name]
#
#     shap_dict = defaultdict(list)
#
#     for item in shap_items:
#         shap_vals = item["shap"]      # (n_samples, n_features)
#         feats = item["features"]
#
#         for i, f in enumerate(feats):
#             shap_dict[f].append(np.abs(shap_vals[:, i]))
#
#     shap_summary = {
#         f: np.mean(np.concatenate(v))
#         for f, v in shap_dict.items()
#     }
#
#     shap_df = (
#         pd.DataFrame(
#             shap_summary.items(),
#             columns=["Feature", "Mean |SHAP value|"]
#         )
#         .sort_values(by="Mean |SHAP value|", ascending=False)
#     )
#
#     # 保存
#     shap_df.to_excel(
#         f"E:/US-MRI/{folder_name}/SHAP_{set_name}_{best_model_name}.xlsx",
#         index=False
#     )
#
#     print(f"[SHAP] {set_name} ({best_model_name}) saved.")


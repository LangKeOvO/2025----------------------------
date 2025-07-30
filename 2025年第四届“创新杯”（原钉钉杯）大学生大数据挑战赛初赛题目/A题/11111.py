import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_score, recall_score, f1_score, roc_curve
from imblearn.combine import SMOTETomek
from xgboost import XGBClassifier
import optuna
import shap
from tqdm import tqdm
import joblib

# 设置Matplotlib以支持中文字符和正确显示负号
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ===================== 1. 读取数据 =====================
print("\n[阶段1] 正在读取数据...")
# 建议使用相对路径或将路径存储在配置文件中，以提高代码的可移植性
file_path = r'D:\桌面\2025年第四届“创新杯”（原钉钉杯）大学生大数据挑战赛初赛题目\2025年第四届“创新杯”（原钉钉杯）大学生大数据挑战赛初赛题目\A题\data\train_data.csv'
df = pd.read_csv(file_path)

# ===================== 2. 预处理与特征工程 =====================
print("\n[阶段2] 正在进行预处理与特征工程...")

# --- 2.1 修正 Installation_Year 并创建 Machine_Age 特征 ---
# 目的：处理原始数据中的异常年份（如未来年份），并将其转换为更有预测价值的“机龄”特征。
print("  [2.1] 修正 Installation_Year 并创建 'Machine_Age' 特征...")
REFERENCE_YEAR = 2025  # 定义一个当前或未来的基准年份
# 识别出无效年份（未来年份或过早的年份）
invalid_years_mask = (df['Installation_Year'] > REFERENCE_YEAR) | (df['Installation_Year'] < 1980)
# 如果存在无效年份，则进行修正
if invalid_years_mask.any():
    # 计算所有有效年份的中位数，用于填充，这比用均值更稳健，不易受极端值影响
    valid_years_median = df.loc[~invalid_years_mask, 'Installation_Year'].median()
    print(f"    发现并修正 {invalid_years_mask.sum()} 个无效安装年份，使用中位数 {valid_years_median} 进行填充。")
    # 定位并替换无效年份
    df.loc[invalid_years_mask, 'Installation_Year'] = valid_years_median
# 计算机器年龄
df['Machine_Age'] = REFERENCE_YEAR - df['Installation_Year']
print("    'Machine_Age' 特征已创建。")


# --- 2.2 特征分箱: Last_Maintenance_Days_Ago ---
# 目的：将连续的维护天数转换为离散的类别，帮助模型捕捉非线性关系。
print("  [2.2] 对 'Last_Maintenance_Days_Ago' 进行特征分箱...")
# 定义分箱的边界，-1确保0也能被正确包含
bins = [-1, 30, 90, np.inf]
# 定义每个箱的标签
labels = ['Maintenance_Recent', 'Maintenance_Medium', 'Maintenance_Long']
# 使用pd.cut进行分箱操作
df['Maintenance_Bin'] = pd.cut(df['Last_Maintenance_Days_Ago'], bins=bins, labels=labels)
print("    'Maintenance_Bin' 特征已创建。")


# --- 2.3 创建交互特征 ---
# 目的：组合现有特征，以发现可能存在的、单个特征无法表达的协同效应。
print("  [2.3] 正在创建交互特征...")
# 功耗与温度的交互，可能反映机器在高负荷下的热状况
df['Power_Temp_Interaction'] = df['Power_Consumption_kW'] * df['Temperature_C']
# 振动与运行小时数的交互，可能反映累积的机械磨损
df['Vibration_Hours_Interaction'] = df['Vibration_mms'] * df['Operational_Hours']
# 维护频率，通过将历史维护次数标准化到机器年龄上，比单纯的维护次数更有信息量
# 分母+1是为了避免机器年龄为0时出现除以零的错误
df['Maintenance_Frequency'] = df['Maintenance_History_Count'] / (df['Machine_Age'] + 1)
print("    'Power_Temp_Interaction', 'Vibration_Hours_Interaction', 'Maintenance_Frequency' 已创建。")


# --- 2.4 缺失值与无用字段处理 ---
print("  [2.4] 正在处理缺失值和删除无用字段...")
# 计算每列的缺失值比例
missing_ratio = df.isnull().mean()
# 删除缺失比例非常高（例如 > 80%）的列，因为它们信息量太少
df.drop(columns=missing_ratio[missing_ratio > 0.8].index, inplace=True)
# 删除已经使用完毕的原始特征、ID以及与目标强相关的标签（防止标签泄漏）
df.drop(columns=['Machine_ID', 'Remaining_Useful_Life_days', 'Installation_Year', 'Last_Maintenance_Days_Ago'], errors='ignore', inplace=True)

# 分离特征矩阵 (X) 和目标向量 (y)
y = df['Failure_Within_7_Days']
X = df.drop(columns=['Failure_Within_7_Days'])

# ===================== 3. 数据划分 =====================
print("\n[阶段3] 正在划分训练集与测试集...")
# stratify=y确保在划分后，训练集和测试集中的故障样本比例与原始数据一致，这对于不平衡学习至关重要
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ===================== 4. 异常值处理（缩尾法） =====================
print("\n[阶段4] 正在处理异常值（缩尾法）...")
# 缩尾法（Winsorizing）是一种处理异常值的方法，它将超出指定分位数范围的极端值替换为该分位数的值
def winsorize_dataframes(X_train_df, X_test_df, lower=0.01, upper=0.99):
    X_train_processed = X_train_df.copy()
    X_test_processed = X_test_df.copy()
    # 只对数值型列进行处理
    numeric_cols = X_train_processed.select_dtypes(include=np.number).columns
    # 遍历所有数值列
    for col in numeric_cols:
        # 在训练集上学习分位数边界，以防数据泄漏
        lower_bound = X_train_processed[col].quantile(lower)
        upper_bound = X_train_processed[col].quantile(upper)
        # 将学习到的边界同时应用于训练集和测试集
        X_train_processed[col] = np.clip(X_train_processed[col], lower_bound, upper_bound)
        X_test_processed[col] = np.clip(X_test_processed[col], lower_bound, upper_bound)
    return X_train_processed, X_test_processed
X_train, X_test = winsorize_dataframes(X_train, X_test)


# ===================== 5. 特征编码（独热编码） =====================
print("\n[阶段5] 正在进行特征编码...")
# 将类别型特征转换为模型可以理解的数值格式
categorical_cols = ['Machine_Type', 'Maintenance_Bin'] 
# 对训练集进行独热编码
X_train = pd.get_dummies(X_train, columns=categorical_cols, drop_first=True)
# 对测试集进行独热编码
X_test = pd.get_dummies(X_test, columns=categorical_cols, drop_first=True)

# 对齐训练集和测试集的列，以处理划分后可能出现的类别不一致问题
train_cols = X_train.columns
# reindex可以保证测试集的列与训练集完全一致，多出的列用0填充，缺少的列被丢弃
X_test = X_test.reindex(columns=train_cols, fill_value=0)


# ===================== 6. 不平衡学习（SMOTETomek） =====================
print("\n[阶段6] 正在进行 SMOTETomek 重采样平衡类别...")
# 只保留fit_resample逻辑，不直接对全训练集重采样，留给CV内处理，避免信息泄露
# 这里仅初始化SMOTETomek对象，后续CV内使用

smt = SMOTETomek(random_state=42)

# ===================== 7. Optuna 超参数调优 =====================
print("\n[阶段7] 正在使用 Optuna 进行超参数调优（XGBoost）...")

def objective(trial):
    params = {
        'max_depth': trial.suggest_int("max_depth", 3, 6),          # 控制树深，降低复杂度
        'learning_rate': trial.suggest_float("learning_rate", 0.01, 0.1),  # 学习率较小，训练稳定
        'n_estimators': trial.suggest_int("n_estimators", 500, 1000),     # 迭代次数适中
        'subsample': trial.suggest_float("subsample", 0.6, 0.9),          # 行采样，防止过拟合
        'colsample_bytree': trial.suggest_float("colsample_bytree", 0.6, 0.9),  # 列采样
        'gamma': trial.suggest_float("gamma", 0, 5),                      # 节点分裂最小损失，剪枝
        'reg_alpha': trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True), # L1正则
        'reg_lambda': trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),# L2正则
        'eval_metric': 'logloss',
        'random_state': 42,
        'n_jobs': -1,
        'use_label_encoder': False
    }

    model = XGBClassifier(**params)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    f1_scores = []

    for train_idx, val_idx in cv.split(X_train, y_train):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        # 仅对训练折做重采样，验证折保持原始分布，防止验证集泄露信息
        X_tr_res, y_tr_res = smt.fit_resample(X_tr, y_tr)

        # 训练时增加early_stopping_rounds防止过拟合
        model.fit(
            X_tr_res, y_tr_res,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=50,
            verbose=False
        )
        y_val_pred = model.predict(X_val)
        f1_scores.append(f1_score(y_val, y_val_pred))

    return np.mean(f1_scores)

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50, show_progress_bar=True)
best_params = study.best_params
print("\n[结果] 最佳参数：", best_params)

# ===================== 8. 训练最终模型 =====================
print("\n[阶段8] 正在使用最佳参数训练最终模型...")
model = XGBClassifier(**best_params, eval_metric='logloss', random_state=42, use_label_encoder=False)

# 对训练集做SMOTETomek重采样
X_train_res, y_train_res = smt.fit_resample(X_train, y_train)

# 训练时拆分一部分作为验证集，用于early stopping
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train_res, y_train_res, test_size=0.2, stratify=y_train_res, random_state=42
)

model.fit(
    X_tr, y_tr,
    eval_set=[(X_val, y_val)],
    early_stopping_rounds=50,
    verbose=True
)

joblib.dump(model, 'xgboost_fault_prediction_model_v2.pkl')
print("模型已保存为 'xgboost_fault_prediction_model_v2.pkl'")


# ===================== 9. 阈值调优与三曲线图 =====================
print("\n[阶段9] 正在绘制阈值-指标曲线并选择最佳阈值...")
def plot_threshold_metrics(model, X_val, y_val):
    # 获取模型对验证集预测为正类（故障）的概率
    y_proba = model.predict_proba(X_val)[:, 1]
    # 定义一系列要测试的阈值
    thresholds = np.linspace(0.1, 0.9, 81)
    precisions, recalls, f1s = [], [], []

    # 遍历所有阈值，计算对应的P, R, F1
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        # zero_division=0 避免在某些极端阈值下（如所有预测都为负）计算指标时因除以零而报错
        precisions.append(precision_score(y_val, y_pred, zero_division=0))
        recalls.append(recall_score(y_val, y_pred, zero_division=0))
        f1s.append(f1_score(y_val, y_pred, zero_division=0))

    # 寻找一个综合性能较好的阈值，这里使用 F1和Recall之和作为标准
    # 在故障预测中，我们通常更关注召回率（不漏报）
    best_idx = np.argmax(np.array(f1s) + np.array(recalls))
    best_threshold = thresholds[best_idx]

    # 绘图
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, precisions, label='精确率 (Precision)')
    plt.plot(thresholds, recalls, label='召回率 (Recall)')
    plt.plot(thresholds, f1s, label='F1 分数 (F1-Score)', linewidth=2.5)
    # 绘制找到的最佳阈值线
    plt.axvline(best_threshold, linestyle='--', color='red', label=f'最优阈值 ≈ {best_threshold:.3f}')
    plt.xlabel("分类概率阈值")
    plt.ylabel("指标值")
    plt.title("模型评估指标与分类阈值的关系")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

    return best_threshold

# 在测试集上寻找最佳阈值
best_threshold = plot_threshold_metrics(model, X_test, y_test)

# ===================== 10. 模型评估 =====================
print(f"\n[阶段10] 正在使用最优阈值 {best_threshold:.4f} 在测试集上评估模型性能...")
# 获取测试集的预测概率
y_proba = model.predict_proba(X_test)[:, 1]
# 使用找到的最佳阈值将概率转换为最终的0/1预测
y_pred = (y_proba >= best_threshold).astype(int)

# 打印各项评估指标
print("分类报告:\n", classification_report(y_test, y_pred, digits=4))
roc_auc = roc_auc_score(y_test, y_proba)
print(f"ROC-AUC: {roc_auc:.4f}")

# 绘制ROC曲线
fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}", color='blue')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='随机猜测')
plt.xlabel("假正率 (FPR)")
plt.ylabel("真正率 (TPR)")
plt.title("ROC 曲线")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# ===================== 11. 特征重要性分析 =====================
print("\n[阶段11] 正在绘制特征重要性图...")
# 从训练好的模型中提取特征重要性
# 注意：index必须使用X_train.columns，因为这是经过独热编码后的最终特征名
importance = pd.Series(model.feature_importances_, index=X_train.columns)
# 排序并选出最重要的前5个特征
top5 = importance.sort_values(ascending=False).head(5)
print("\n前五个重要特征:\n", top5)

# 绘制条形图
plt.figure(figsize=(9, 6))
top5.plot(kind='barh', color='teal')
plt.title("特征重要性 (Top 5)")
plt.xlabel("重要性得分 (Importance Score)")
# 将最重要的特征显示在顶部
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# ===================== 12. 混淆矩阵 =====================
print("\n[阶段12] 正在绘制混淆矩阵...")
# 计算混淆矩阵
cm = confusion_matrix(y_test, y_pred)
# 使用seaborn的热力图进行可视化
plt.figure(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title("混淆矩阵")
plt.xlabel("预测标签")
plt.ylabel("真实标签")
plt.tight_layout()
plt.show()

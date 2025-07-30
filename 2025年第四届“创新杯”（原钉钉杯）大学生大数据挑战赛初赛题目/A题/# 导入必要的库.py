import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.inspection import PartialDependenceDisplay, permutation_importance
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import optuna
import shap
import matplotlib.pyplot as plt

RANDOM_STATE = 42
def tune_xgb(X_train, y_train, cv, n_trials=50):
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "verbosity": 0,
            "use_label_encoder": False,
            "random_state": RANDOM_STATE
        }
        model = XGBRegressor(**params)
       
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="neg_mean_squared_error")
        rmse = np.sqrt(-scores.mean())
        return rmse
   
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, n_jobs=1)
    return study.best_params, study.best_value

def tune_lgb(X_train, y_train, cv, n_trials=50):
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "num_leaves": trial.suggest_int("num_leaves", 20, 300),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "random_state": RANDOM_STATE,
            "verbose": -1
        }
        model = LGBMRegressor(**params)
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="neg_mean_squared_error")
        rmse = np.sqrt(-scores.mean())
        return rmse
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, n_jobs=1)
    return study.best_params, study.best_value

def tune_cat(X_train, y_train, cv, n_trials=50):
    def objective(trial):
        params = {
            "iterations": trial.suggest_int("iterations", 100, 1000),
            "depth": trial.suggest_int("depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.5, 1.0),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 100),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-3, 10.0, log=True),
            "random_state": RANDOM_STATE,
        }
        model = CatBoostRegressor(**params, silent=True)
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="neg_mean_squared_error")
        rmse = np.sqrt(-scores.mean())
        return rmse
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, n_jobs=1)
    return study.best_params, study.best_value


if __name__ == "__main__":
    data = pd.read_csv("train_data.csv")
    X = data.drop("Remaining_Useful_Life_days", axis=1)
non_numeric_cols = X.select_dtypes(include=['object']).columns
print("非数值列：", non_numeric_cols)
if 'MachineID' in non_numeric_cols:
    X = X.drop(columns=['MachineID'])
    non_numeric_cols = non_numeric_cols.drop('MachineID')
if len(non_numeric_cols) > 0:
    from sklearn.preprocessing import LabelEncoder
non_numeric_cols = X.select_dtypes(include='object').columns.tolist()
for col in non_numeric_cols:
    unique_vals = X[col].nunique()
    
    if unique_vals < 100:  
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
    else:
        print(f"⚠️ 列 {col} 唯一值过多（{unique_vals}），已删除以节省内存。")
        X.drop(columns=col, inplace=True)

feature_names = list(X.columns)
y = data["Remaining_Useful_Life_days"]
feature_names = list(X.columns)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
best_params_xgb, best_val_xgb = tune_xgb(X_train, y_train, cv)
print("XGBoost最佳超参数:", best_params_xgb)
print(f"XGBoost调参所得最优RMSE: {best_val_xgb:.4f}")
model_xgb = XGBRegressor(**best_params_xgb, random_state=RANDOM_STATE, verbosity=0, use_label_encoder=False)
model_xgb.fit(X_train, y_train)
best_params_lgb, best_val_lgb = tune_lgb(X_train, y_train, cv)
print("LightGBM最佳超参数:", best_params_lgb)
print(f"LightGBM调参所得最优RMSE: {best_val_lgb:.4f}")
model_lgb = LGBMRegressor(**best_params_lgb, random_state=RANDOM_STATE, verbose=-1)
model_lgb.fit(X_train, y_train)
best_params_cat, best_val_cat = tune_cat(X_train, y_train, cv)
print("CatBoost最佳超参数:", best_params_cat)
print(f"CatBoost调参所得最优RMSE: {best_val_cat:.4f}")
model_cat = CatBoostRegressor(**best_params_cat, random_state=RANDOM_STATE, silent=True)
model_cat.fit(X_train, y_train)
y_pred_xgb = model_xgb.predict(X_test)
y_pred_lgb = model_lgb.predict(X_test)
y_pred_cat = model_cat.predict(X_test)
mse_xgb = mean_squared_error(y_test, y_pred_xgb); rmse_xgb = np.sqrt(mse_xgb); r2_xgb = r2_score(y_test, y_pred_xgb)
mse_lgb = mean_squared_error(y_test, y_pred_lgb); rmse_lgb = np.sqrt(mse_lgb); r2_lgb = r2_score(y_test, y_pred_lgb)
mse_cat = mean_squared_error(y_test, y_pred_cat); rmse_cat = np.sqrt(mse_cat); r2_cat = r2_score(y_test, y_pred_cat)
print(f"XGBoost - MSE: {mse_xgb:.4f}, RMSE: {rmse_xgb:.4f}, R^2: {r2_xgb:.4f}")
print(f"LightGBM - MSE: {mse_lgb:.4f}, RMSE: {rmse_lgb:.4f}, R^2: {r2_lgb:.4f}")
print(f"CatBoost - MSE: {mse_cat:.4f}, RMSE: {rmse_cat:.4f}, R^2: {r2_cat:.4f}")
best_model = None
best_model_name = None
lowest_rmse = float("inf")
if rmse_xgb < lowest_rmse:
        best_model = model_xgb
        best_model_name = "XGBoost"
        lowest_rmse = rmse_xgb
if rmse_lgb < lowest_rmse:
        best_model = model_lgb
        best_model_name = "LightGBM"
        lowest_rmse = rmse_lgb
if rmse_cat < lowest_rmse:
        best_model = model_cat
        best_model_name = "CatBoost"
        lowest_rmse = rmse_cat
print(f"最佳模型: {best_model_name} (测试集RMSE={lowest_rmse:.4f})")

    # 可视化1: 预测值 vs 实际值 散点图
plt.figure(figsize=(6, 5))
plt.scatter(y_test, best_model.predict(X_test), alpha=0.7)
    # 绘制y=x参考线
line_min = min(y_test.min(), best_model.predict(X_test).min())
line_max = max(y_test.max(), best_model.predict(X_test).max())
plt.plot([line_min, line_max], [line_min, line_max], 'r--')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title(f'Actual vs Predicted ({best_model_name})')
plt.show()

    # 可视化2: 残差分析图（预测值 vs 残差）
residuals = best_model.predict(X_test) - y_test
plt.figure(figsize=(6, 5))
plt.scatter(best_model.predict(X_test), residuals, alpha=0.7)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Predicted')
plt.ylabel('Residual')
plt.title(f'Residuals vs Predicted ({best_model_name})')
plt.show()

    # 可视化3: 模型特征重要性（基于模型内部指标，例如Gain值）
importances = best_model.feature_importances_
indices = np.argsort(importances)[::-1]
plt.figure(figsize=(8, 6))
plt.barh(range(len(importances)), importances[indices])
plt.yticks(range(len(importances)), [feature_names[i] for i in indices])
plt.xlabel('Feature Importance')
plt.title(f'Feature Importance ({best_model_name})')
plt.gca().invert_yaxis()  # 将最高重要度特征放顶部
plt.show()

    # 可视化4: 部分依赖图 (Partial Dependence) - 对最佳模型最重要的三个特征绘制PDP3
top_features = indices[:3] if len(indices) >= 3 else indices
PartialDependenceDisplay.from_estimator(best_model, X_train, top_features, feature_names=feature_names, grid_resolution=20)
plt.show()

    # 可视化5: Permutation特征重要性（通过打乱特征衡量性能下降）
result = permutation_importance(best_model, X_test, y_test, n_repeats=10, scoring='neg_mean_squared_error', random_state=RANDOM_STATE)
perm_importances = result.importances_mean
perm_indices = np.argsort(perm_importances)[::-1]
plt.figure(figsize=(8, 6))
plt.barh(range(len(perm_importances)), perm_importances[perm_indices], xerr=result.importances_std[perm_indices], align='center')
plt.yticks(range(len(perm_importances)), [feature_names[i] for i in perm_indices])
plt.xlabel('Permutation Importance (drop in performance)')
plt.title(f'Permutation Feature Importance ({best_model_name})')
plt.gca().invert_yaxis()
plt.show()

    # 可视化6: SHAP特征重要性（基于SHAP值的特征影响力）
explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_test)
    # 绘制SHAP值总结图（特征影响总体排序）4
shap.summary_plot(shap_values, X_test, feature_names=feature_names)
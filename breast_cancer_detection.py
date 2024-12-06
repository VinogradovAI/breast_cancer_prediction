import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, roc_curve, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
from catboost import CatBoostClassifier
import lightgbm as lgb
import matplotlib.pyplot as plt

# Создание директорий для сохранения результатов
os.makedirs("results/plots", exist_ok=True)
os.makedirs("results/tables", exist_ok=True)

from warnings import filterwarnings
filterwarnings("ignore")

# Функция для предобработки данных с выделением валидационной выборки
def preprocess_data_with_validation(df, target_column, val_size=0.2, test_size=0.2):
    """
    Предобработка данных с выделением валидационной выборки.
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Разделяем данные на тренировочную + валидационную выборки и тестовую
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Разделяем тренировочную выборку на тренировочную и валидационную
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=val_size, random_state=42)

    # Масштабируем данные
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    return X_train, X_val, X_test, y_train, y_val, y_test


# Функция для оценки модели
def evaluate_model(model, X, y):
    """
    Обучение модели и расчет метрик.
    """
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else None
    accuracy = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    auc = roc_auc_score(y, y_prob) if y_prob is not None else None
    return accuracy, f1, auc, y_prob


# Функция для построения ROC-кривой
def plot_roc_curve(y, y_prob, model_name, dataset_name, stage):
    """
    Построение ROC-кривой для модели.
    """
    fpr, tpr, _ = roc_curve(y, y_prob)
    auc_score = roc_auc_score(y, y_prob)
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, label=f'AUC = {auc_score:.2f}', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
    plt.title(f"ROC Curve - {model_name} ({dataset_name}, {stage})", fontsize=14)
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid()
    plt.savefig(f"results/plots/roc_{model_name}_{dataset_name}_{stage}.png")
    plt.close()


# Функция для построения графика важности признаков
def plot_feature_importance(model, feature_names, model_name, dataset_name):
    """
    Построение графика важности признаков для модели.
    """
    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
        sorted_idx = np.argsort(importance)[::-1]
        sorted_importance = importance[sorted_idx]
        sorted_features = feature_names[sorted_idx]
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(sorted_importance)), sorted_importance, align="center")
        plt.xticks(range(len(sorted_importance)), sorted_features, rotation=45, ha='right', fontsize=10)
        plt.title(f"Feature Importance - {model_name} ({dataset_name})", fontsize=14)
        plt.ylabel("Importance", fontsize=12)
        plt.grid()
        plt.tight_layout()
        plt.savefig(f"results/plots/feature_importance_{model_name}_{dataset_name}.png")
        plt.close()


# Функция для подбора гиперпараметров
def tune_hyperparameters(model, param_grid, X_train, y_train):
    """
    Подбор гиперпараметров с помощью GridSearchCV.
    """
    grid_search = GridSearchCV(model, param_grid, scoring='f1', cv=5, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_


# Функция для обработки одного датасета
def process_dataset_with_validation(df, target_column, dataset_name):
    """
    Обработка датасета с учетом валидационной выборки.
    """
    X_train, X_val, X_test, y_train, y_val, y_test = preprocess_data_with_validation(df, target_column)
    feature_names = df.drop(columns=[target_column]).columns

    models = {
        "Logistic Regression": (LogisticRegression(), {}),
        "Random Forest": (RandomForestClassifier(), {"n_estimators": [50, 100], "max_depth": [5, 10]}),
        "Gradient Boosting": (GradientBoostingClassifier(), {"n_estimators": [50, 100], "learning_rate": [0.01, 0.1]}),
        "SVM": (SVC(probability=True), {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"]}),
        "Neural Network": (MLPClassifier(max_iter=500), {"hidden_layer_sizes": [(50,), (100,)]}),
        "XGBoost": (xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss"), {"n_estimators": [50, 100]}),
        "CatBoost": (CatBoostClassifier(verbose=0), {"depth": [6, 8], "iterations": [50, 100]}),
        "LightGBM": (lgb.LGBMClassifier(), {"n_estimators": [50, 100], "num_leaves": [20, 31]})
    }

    results = []

    for model_name, (model, param_grid) in models.items():
        if param_grid:
            tuned_model = tune_hyperparameters(model, param_grid, X_train, y_train)
        else:
            tuned_model = model
            tuned_model.fit(X_train, y_train)  # Ensure the model is fitted

        # Оценка на валидационной выборке
        val_accuracy, val_f1, val_auc, val_y_prob = evaluate_model(tuned_model, X_val, y_val)

        # Итоговая оценка на тестовой выборке
        test_accuracy, test_f1, test_auc, test_y_prob = evaluate_model(tuned_model, X_test, y_test)

        # Сохранение результатов
        results.append({
            "Model": model_name,
            "Val Accuracy": val_accuracy,
            "Val F1": val_f1,
            "Val AUC": val_auc,
            "Test Accuracy": test_accuracy,
            "Test F1": test_f1,
            "Test AUC": test_auc
        })

        # Построение ROC-кривых
        if val_y_prob is not None:
            plot_roc_curve(y_val, val_y_prob, model_name, dataset_name, "Validation")
        if test_y_prob is not None:
            plot_roc_curve(y_test, test_y_prob, model_name, dataset_name, "Test")

        # Построение графика важности признаков
        if hasattr(tuned_model, "feature_importances_"):
            plot_feature_importance(tuned_model, feature_names, model_name, dataset_name)

    # Сохранение результатов в таблицу
    results_df = pd.DataFrame(results)
    results_df.to_csv(f"results/tables/results_{dataset_name}.csv", index=False)
    return results_df


# Основной код для обработки всех датасетов

TARGET_VAR = 'Diagnosis'

DATA_PROCEED_PATH = "D:/ml_school/Skillfactory/team_cases/breast_cancer_prediction/data/processed/"
datasets = ["dataset1_processed.csv", "dataset2_top_features.csv", "dataset3_pca.csv"]

final_results = []

for i, dataset_path in enumerate(datasets):
    df = pd.read_csv(DATA_PROCEED_PATH + dataset_path)
    dataset_name = f"Dataset_{i + 1}"
    results_df = process_dataset_with_validation(df, target_column=TARGET_VAR, dataset_name=dataset_name)
    final_results.append(results_df)

# Сводная таблица
final_summary = pd.concat(final_results, keys=[f"Dataset_{i + 1}" for i in range(len(datasets))])
final_summary.to_csv("results/tables/final_summary.csv")

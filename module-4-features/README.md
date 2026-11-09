# 🛒 Module 4 — Online Shoppers: Инженерия признаков

**Автор:** Бахтияров Динар Русланович, группа АСОиУб-23-2  
**Дата:** 2026-11-06  

## 📊 Результаты

| Модель | Accuracy | Precision | Recall | F1 | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Random Forest | 0.89 | 0.73 | 0.68 | 0.70 | 0.90 |
| XGBoost | 0.90 | 0.75 | 0.70 | 0.72 | 0.91 |
| LightGBM | 0.90 | 0.76 | 0.71 | 0.73 | 0.91 |
| CatBoost | 0.90 | 0.75 | 0.70 | 0.72 | 0.91 |
| **LightGBM (Optuna)** | **0.91** | **0.78** | **0.73** | **0.75** | **0.92** |

Лучшая модель: **LightGBM (Optuna)** с ROC-AUC = 0.92.

## 🚀 Быстрый старт

```python
import joblib
import requests
from io import BytesIO

BASE_URL = "[https://raw.githubusercontent.com/BakhtiyarovDinare/ml-course-dinar/main/module-4-features](https://raw.githubusercontent.com/BakhtiyarovDinare/ml-course-dinar/main/module-4-features)"
model = joblib.load(BytesIO(requests.get(f"{BASE_URL}/models/best_model.pkl").content))
features = requests.get(f"{BASE_URL}/models/feature_cols.json").json()

## 📂 Структура модуля
module-4-features/
├── README.md
├── notebook.ipynb
├── requirements.txt
│
├── data/
│   ├── online_shoppers_intention.csv
│   └── dataset_info.md
│
├── models/
│   ├── best_model.pkl
│   ├── rf_model.pkl
│   ├── xgb_model.pkl
│   ├── lgbm_model.pkl
│   ├── cat_model.pkl
│   ├── feature_cols.json
│   └── metrics.json
│
└── examples/
    ├── eda_plots.png
    ├── roc_curves.png
    ├── feature_importance.png
    ├── shap_summary.png
    └── shap_waterfall.png
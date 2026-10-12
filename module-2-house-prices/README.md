# Module 2 — House Prices: предобработка и регрессия

**Автор:** Бахтияров Динар Русланович, группа АСОиУб-23-2  
**Дата:** 2026-10-13  

## Результаты

| Модель | RMSE (log) | MAE (log) | R² (log) | RMSE ($) | R² (real) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Linear Regression** | 0.1251 | 0.0891 | 0.8944 | 22691 | 0.8978 |
| **Ridge** | 0.1246 | 0.0887 | 0.8953 | 22628 | 0.8984 |
| **Lasso** | 0.1222 | 0.0850 | 0.8992 | 22031 | 0.9037 |
| **ElasticNet** | 0.1223 | 0.0850 | 0.8991 | 21996 | 0.9040 |

**Время обучения:** < 0.5 сек на каждую модель  
**Лучшая модель:** Lasso ($R^2 = 0.9037$)  

## 📁 Структура модуля

    module-2-house-prices/
    ├── README.md           ← этот файл
    ├── notebook.ipynb      ← полный отчёт: 14 разделов
    ├── requirements.txt    ← зависимости
    ├── data/
    │   ├── train.csv       ← 1460 строк с целевой переменной
    │   ├── test.csv        ← 1459 строк для submission
    │   └── house_prices_info.md ← описание датасета
    ├── models/             ← сохранённые модели и метаданные (.pkl, .json)
    └── examples/           ← PNG-визуализации

## 🚀 Быстрый старт

Пример загрузки лучшей модели из репозитория на Python:

```python
import joblib
import requests
from io import BytesIO

BASE_URL = "[https://raw.githubusercontent.com/BakhtiyarovDinar/ml-course-dinar/main/module-2-house-prices](https://raw.githubusercontent.com/BakhtiyarovDinar/ml-course-dinar/main/module-2-house-prices)"

model = joblib.load(BytesIO(requests.get(f"{BASE_URL}/models/lasso_model.pkl").content))
scaler = joblib.load(BytesIO(requests.get(f"{BASE_URL}/models/scaler.pkl").content))
print("✅ Модель и scaler успешно загружены из репозитория!")
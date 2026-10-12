# Module 1 — Titanic: EDA и бинарная классификация

**Автор:** Бахтияров Динар Русланович, группа АСОиУб-23-2  
**Дисциплина:** Машинное обучение и искусственный интеллект  
**Дата сдачи:** 2026-09-29  

## 📊 Результаты

| Модель | Accuracy | Precision | Recall | F1 | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.8045 | 0.7833 | 0.6812 | 0.7287 | 0.8486 |
| **Decision Tree** | 0.7821 | 0.7419 | 0.6667 | 0.7023 | 0.8132 |

**Лучшая модель:** Logistic Regression, ROC-AUC = 0.8486.  

## 🎯 Ключевые инсайты

- Женщины выживали в значительно большей пропорции, чем мужчины.
- Пассажиры 1-го класса имели наивысший приоритет при эвакуации по сравнению с 2-м и 3-м классами.
- Возраст и стоимость билета (Fare) оказали ключевое влияние на качество прогнозирования моделей.

## 📁 Структура модуля

```text
module-1-titanic/
├── README.md           # этот файл
├── notebook.ipynb      # полный отчёт: 16 разделов
├── requirements.txt    # зависимости
├── data/
│   ├── train.csv       # обучающая выборка
│   └── test.csv        # тестовая выборка
├── models/             # сохранённые модели и метаданные (.pkl, .json)
└── examples/           # PNG-визуализации (графики EDA, ROC, матрицы ошибок)

## Быстрый старт

```python
import joblib
import requests
from io import BytesIO

BASE_URL = "https://raw.githubusercontent.com/BakhtiyarovDinar/ml-course-dinar/main/module-1-titanic"

model = joblib.load(BytesIO(requests.get(f"{BASE_URL}/models/lr_model.pkl").content))
scaler = joblib.load(BytesIO(requests.get(f"{BASE_URL}/models/scaler.pkl").content))
metrics = requests.get(f"{BASE_URL}/models/metrics.json").json()

print("Метрики сохраненных моделей загружены успешно!")


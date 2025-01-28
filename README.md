# Sentiment-Topic Classifier

Проект для анализа тональности и классификации тем текстов на русском языке. Использует современные методы обработки естественного языка и глубокого обучения.

## Описание проекта

Классификатор выполняет две задачи:
1. Определение тональности текста (позитивная/нейтральная/негативная)
2. Классификация темы текста

## Технологии

- Python 3.8+
- PyTorch
- Transformers (Hugging Face)
- BERT (DeepPavlov/rubert-base-cased)
- Pandas, NumPy
- Scikit-learn

## Структура проекта

```
sentiment-topic-classifier/
├── src/
│   ├── data/           # Обработка данных
│   ├── models/         # Модели
│   ├── training/       # Логика обучения
│   └── utils/          # Вспомогательные функции
├── configs/            # Конфигурационные файлы
├── notebooks/          # Jupyter notebooks
└── requirements.txt    # Зависимости
```

## Установка

```bash
# Клонирование репозитория
git clone https://github.com/your-username/sentiment-topic-classifier.git
cd sentiment-topic-classifier

# Установка зависимостей
pip install -r requirements.txt
```

## Использование

```python
from src.models.classifier import MultiTaskClassifier
from transformers import AutoTokenizer

# Загрузка модели
model = MultiTaskClassifier(
    model_name="DeepPavlov/rubert-base-cased",
    num_sentiment_labels=3,
    num_topic_labels=8
)

# Пример использования
text = "Ваш текст для анализа"
tokenizer = AutoTokenizer.from_pretrained("DeepPavlov/rubert-base-cased")
```

## Обучение модели

1. Подготовка данных
2. Настройка параметров в configs/config.yaml
3. Запуск обучения

## Метрики качества

- Accuracy
- F1-score
- Confusion Matrix

## Лицензия

MIT
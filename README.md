# Конвейер прогнозирования оттока абонентов (Churn Prediction Pipeline)

Модульный, воспроизводимый и идемпотентный ML-конвейер для решения задачи бинарной классификации оттока абонентов (`churned`). Конвейер загружает предобработанные признаки и целевую разметку, выполняет очистку и масштабирование данных, обучает модель `RandomForestClassifier`, оценивает качество на валидационной и тестовой выборках и сохраняет артефакты для последующего инференса.

Подробная схема конвейера: [architecture.md](architecture.md) (графическая версия — [architecture.png](architecture.png)).

## Структура проекта

```text
.
├── src/
│   └── pipeline.py          # Основной модуль конвейера
├── config.json              # Параметры путей, разбиения, модели и логирования
├── requirements.txt         # Зависимости Python
├── Dockerfile               # Описание Docker-образа
├── .dockerignore            # Исключения для контекста сборки Docker
├── .gitignore               # Исключения для Git
├── architecture.md          # Архитектурная схема (Mermaid + описание слоёв)
├── architecture.png         # Графическая схема конвейера
├── refactoring_report.md    # Отчёт по проведённому рефакторингу
├── data/                    # Входные данные (features.parquet, churn_truth.csv)
├── artifacts/model/         # Выходные артефакты модели
└── logs/                    # Файлы логов выполнения
```

## Установка зависимостей

Требуется Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Подготовка данных

Поместите входные файлы в директорию `data/` (либо укажите собственные пути в `config.json`):

- `data/features.parquet` — витрина признаков с ключом `subscriber_key`;
- `data/churn_truth.csv` — разметка целевой переменной с колонками `subscriber_key` и `churned`.

## Запуск конвейера

### Локальный запуск

```bash
python src/pipeline.py --config config.json --verify
```

Флаг `--verify` опционально запускает проверку загрузки сохранённых артефактов и тестового инференса на первых 100 строках `features.parquet`.

### Запуск в Docker

1. Сборка образа:
```bash
docker build -t churn-pipeline:latest .
```

2. Запуск контейнера с монтированием данных, артефактов и логов:
```bash
docker run --rm \
  -v "$(pwd)/data:/app/data:ro" \
  -v "$(pwd)/artifacts:/app/artifacts" \
  -v "$(pwd)/logs:/app/logs" \
  churn-pipeline:latest
```

## Перечень артефактов

После выполнения конвейера создаются следующие файлы (пути задаются в `config.json`):

| Артефакт | Путь по умолчанию | Назначение |
|---|---|---|
| Обученная модель | `artifacts/model/best_model.pkl` | Сериализованный `RandomForestClassifier` (`joblib`) |
| Масштабировщик | `artifacts/model/scaler.pkl` | Обученный на `train`-выборке `StandardScaler` (`joblib`) |
| Список признаков | `artifacts/model/features.json` | Упорядоченный перечень числовых признаков, ожидаемых моделью |
| Метрики качества | `artifacts/model/metrics.json` | Значения ROC-AUC, F1, Precision, Recall (val/test) и время обучения |
| Журнал выполнения | `logs/pipeline.log` | Логи всех этапов конвейера и ошибок |

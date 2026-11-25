# Архитектура ML-конвейера прогнозирования оттока

Графическая версия схемы также доступна в файле [architecture.png](architecture.png).

## Схема конвейера

```mermaid
flowchart TB
    subgraph Config["Слой конфигурации"]
        CFG["config.json<br/>(пути, сплиты, гиперпараметры,<br/>настройки логирования)"]
    end

    subgraph Data["Слой источников данных (data/)"]
        F["features.parquet<br/>(витрина признаков)"]
        C["churn_truth.csv<br/>(разметка: subscriber_key, churned)"]
    end

    subgraph Pipeline["Слой обработки и обучения (src/pipeline.py / Docker)"]
        L["1. load_data()<br/>Валидация файлов, отбор id+target,<br/>inner join по subscriber_key"]
        P["2. preprocess()<br/>Отбор числовых признаков, inf/NaN -&gt; 0,<br/>stratified split 60/20/20, StandardScaler"]
        T["3. train_model()<br/>RandomForestClassifier<br/>+ замер времени обучения"]
        E["4. evaluate()<br/>ROC-AUC, F1, Precision, Recall<br/>на val и test"]
        S["5. save_artifacts()<br/>Удаление старых + атомарная<br/>запись новых артефактов"]
        V["6. verify_artifacts() (опц.)<br/>Smoke-тест загрузки и инференса"]
    end

    subgraph Storage["Слой хранения артефактов (artifacts/model/)"]
        M["best_model.pkl"]
        SC["scaler.pkl"]
        FE["features.json"]
        ME["metrics.json"]
    end

    subgraph Obs["Слой наблюдаемости (logs/)"]
        LOG["pipeline.log<br/>(FileHandler + StreamHandler)"]
    end

    F --> L
    C --> L
    CFG -.-> L
    CFG -.-> P
    CFG -.-> T
    CFG -.-> S
    CFG -.-> LOG

    L -->|"df_model"| P
    P -->|"X_train, y_train"| T
    P -->|"X_val, y_val, X_test, y_test"| E
    P -->|"scaler, feature_cols"| S
    T -->|"model, train_time"| E
    T -->|"model"| S
    E -->|"metrics"| S

    S --> M
    S --> SC
    S --> FE
    S --> ME

    M -.-> V
    SC -.-> V
    FE -.-> V
    F -.-> V

    L -.-> LOG
    P -.-> LOG
    T -.-> LOG
    E -.-> LOG
    S -.-> LOG
    V -.-> LOG
```

## Слои и зависимости между этапами

| Слой | Компоненты | Входы | Выходы / Хранилище |
|---|---|---|---|
| **Конфигурация** | `config.json` | Параметры запуска | Управляет всеми этапами конвейера без правки кода |
| **Источники данных** | `features.parquet`, `churn_truth.csv` | Внешняя выгрузка | `data/` (монтируется в Docker в режиме `read-only`) |
| **Загрузка данных** | `load_data()` | Пути и имена ключевых колонок из `config.json` | Объединённый `DataFrame` (`df_model`) без служебных колонок разметки |
| **Препроцессинг** | `preprocess()` | `df_model`, параметры `preprocessing` | `X_train/val/test`, `y_train/val/test`, `feature_cols`, обученный `StandardScaler` |
| **Обучение** | `train_model()` | `X_train`, `y_train`, гиперпараметры `model` | Обученный `RandomForestClassifier`, `train_time_sec` |
| **Оценка качества** | `evaluate()` | `model`, `X_val/test`, `y_val/test` | Словарь метрик `metrics` (ROC-AUC, F1, Precision, Recall) |
| **Хранение артефактов** | `save_artifacts()` | `model`, `scaler`, `feature_cols`, `metrics` | `artifacts/model/{best_model.pkl, scaler.pkl, features.json, metrics.json}` |
| **Наблюдаемость** | `setup_logging()` | Секция `logging` в `config.json` | `logs/pipeline.log` и вывод в `stdout` |

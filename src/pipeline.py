import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_config(config_path: str) -> Dict[str, Any]:
    """Загружает конфигурацию конвейера из JSON-файла."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Файл конфигурации не найден: {config_path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def setup_logging(logging_cfg: Dict[str, Any]) -> logging.Logger:
    """Настраивает логирование в консоль и файл с перезаписью при каждом запуске."""
    log_file = Path(logging_cfg.get("log_file", "logs/pipeline.log"))
    log_file.parent.mkdir(parents=True, exist_ok=True)

    level_name = str(logging_cfg.get("level", "INFO")).upper()
    level = getattr(logging, level_name, logging.INFO)
    log_format = logging_cfg.get("format", "%(asctime)s - %(levelname)s - %(message)s")

    logging.basicConfig(
        level=level,
        format=log_format,
        handlers=[
            logging.FileHandler(log_file, mode="w", encoding="utf-8"),
            logging.StreamHandler(),
        ],
        force=True,
    )
    return logging.getLogger("churn_pipeline")


def load_data(
    features_path: str,
    churn_path: str,
    id_col: str = "subscriber_key",
    target_col: str = "churned",
) -> pd.DataFrame:
    """
    Загружает таблицу признаков (Parquet) и целевую переменную (CSV),
    объединяя их по ключу абонента.
    Из файла разметки берутся только id_col и target_col для исключения утечек.
    """
    logger = logging.getLogger("churn_pipeline")
    try:
        logger.info("Проверка входных файлов: %s, %s", features_path, churn_path)
        if not os.path.exists(features_path):
            raise FileNotFoundError(f"Файл с признаками не найден: {features_path}")
        if not os.path.exists(churn_path):
            raise FileNotFoundError(f"Файл с целевой переменной не найден: {churn_path}")

        features_df = pd.read_parquet(features_path)
        churn_df = pd.read_csv(churn_path)

        if id_col not in features_df.columns:
            raise KeyError(f"Колонка '{id_col}' отсутствует в {features_path}")
        for required_col in (id_col, target_col):
            if required_col not in churn_df.columns:
                raise KeyError(f"Колонка '{required_col}' отсутствует в {churn_path}")

        # Отбираем только ключ и таргет, чтобы вспомогательные поля (например, churn_day_index)
        # не попали в матрицу признаков и не вызвали утечку данных.
        churn_clean = churn_df[[id_col, target_col]].drop_duplicates(subset=[id_col])
        df_model = features_df.merge(churn_clean, on=id_col, how="inner")

        logger.info(
            "Данные загружены и объединены. Размер датасета: %s, доля целевого класса: %.4f",
            df_model.shape,
            float(df_model[target_col].mean()),
        )
        return df_model
    except Exception as exc:
        logger.error("Ошибка при загрузке данных: %s", exc, exc_info=True)
        raise


def preprocess(
    df_model: pd.DataFrame,
    preprocessing_cfg: Dict[str, Any],
    id_col: str = "subscriber_key",
    target_col: str = "churned",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.Series, pd.Series, pd.Series, List[str], StandardScaler]:
    """
    Отбирает числовые признаки, обрабатывает inf/NaN, стратифицированно делит выборку
    на train/val/test и масштабирует признаки (fit только на train).
    """
    logger = logging.getLogger("churn_pipeline")
    exclude_cols = {id_col, target_col}
    numeric_cols = df_model.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [col for col in numeric_cols if col not in exclude_cols]

    if not feature_cols:
        raise ValueError("После фильтрации не осталось числовых признаков для обучения.")

    X = df_model[feature_cols].replace([np.inf, -np.inf], 0).fillna(0)
    y = df_model[target_col]

    test_size = float(preprocessing_cfg.get("test_size", 0.4))
    val_size = float(preprocessing_cfg.get("val_size", 0.5))
    random_state = int(preprocessing_cfg.get("random_state", 42))

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=val_size, random_state=random_state, stratify=y_temp
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    logger.info(
        "Препроцессинг завершён. Признаков: %d | Train: %d, Val: %d, Test: %d",
        len(feature_cols),
        len(X_train),
        len(X_val),
        len(X_test),
    )
    return (
        X_train_scaled,
        X_val_scaled,
        X_test_scaled,
        y_train,
        y_val,
        y_test,
        feature_cols,
        scaler,
    )


def train_model(
    X_train: np.ndarray,
    y_train: pd.Series,
    model_cfg: Dict[str, Any],
) -> Tuple[RandomForestClassifier, float]:
    """Обучает RandomForestClassifier с параметрами из конфигурации и замеряет время."""
    logger = logging.getLogger("churn_pipeline")
    model_params = {
        "n_estimators": 100,
        "max_depth": 10,
        "class_weight": "balanced",
        "random_state": 42,
        "n_jobs": -1,
    }
    if model_cfg:
        model_params.update(model_cfg)

    logger.info("Запуск обучения RandomForestClassifier с параметрами: %s", model_params)
    model = RandomForestClassifier(**model_params)

    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time

    logger.info("Обучение завершено за %.2f сек.", train_time)
    return model, train_time


def evaluate(
    model: RandomForestClassifier,
    X_val: np.ndarray,
    y_val: pd.Series,
    X_test: np.ndarray,
    y_test: pd.Series,
) -> Dict[str, float]:
    """Вычисляет метрики качества (ROC-AUC, F1, Precision, Recall) на val и test."""
    logger = logging.getLogger("churn_pipeline")

    val_proba = model.predict_proba(X_val)[:, 1]
    val_pred = (val_proba >= 0.5).astype(int)

    test_proba = model.predict_proba(X_test)[:, 1]
    test_pred = (test_proba >= 0.5).astype(int)

    metrics = {
        "val_auc": float(roc_auc_score(y_val, val_proba)),
        "val_f1": float(f1_score(y_val, val_pred, zero_division=0)),
        "val_precision": float(precision_score(y_val, val_pred, zero_division=0)),
        "val_recall": float(recall_score(y_val, val_pred, zero_division=0)),
        "test_auc": float(roc_auc_score(y_test, test_proba)),
        "test_f1": float(f1_score(y_test, test_pred, zero_division=0)),
        "test_precision": float(precision_score(y_test, test_pred, zero_division=0)),
        "test_recall": float(recall_score(y_test, test_pred, zero_division=0)),
    }
    logger.info("Метрики качества: %s", json.dumps(metrics, ensure_ascii=False))
    return metrics


def _atomic_joblib_dump(obj: Any, target_path: Path) -> None:
    """Атомарно сохраняет бинарный артефакт через временный файл."""
    tmp_path = target_path.with_suffix(target_path.suffix + ".tmp")
    joblib.dump(obj, tmp_path)
    os.replace(tmp_path, target_path)


def _atomic_json_dump(obj: Any, target_path: Path) -> None:
    """Атомарно сохраняет JSON-артефакт через временный файл."""
    tmp_path = target_path.with_suffix(target_path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, target_path)


def save_artifacts(
    model: RandomForestClassifier,
    scaler: StandardScaler,
    feature_cols: List[str],
    metrics: Dict[str, float],
    artifacts_cfg: Dict[str, Any],
) -> None:
    """
    Идемпотентно сохраняет артефакты конвейера:
    создаёт директорию, удаляет устаревшие файлы и атомарно записывает новые.
    """
    logger = logging.getLogger("churn_pipeline")
    model_dir = Path(artifacts_cfg["model_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Директория артефактов готова: %s", model_dir)

    paths = {
        "model": model_dir / artifacts_cfg["model_name"],
        "scaler": model_dir / artifacts_cfg["scaler_name"],
        "features": model_dir / artifacts_cfg["features_name"],
        "metrics": model_dir / artifacts_cfg["metrics_name"],
    }

    for name, path in paths.items():
        if path.exists():
            path.unlink()
            logger.info("Удалён предыдущий артефакт (%s): %s", name, path)

    _atomic_joblib_dump(model, paths["model"])
    logger.info("Модель сохранена: %s", paths["model"])

    _atomic_joblib_dump(scaler, paths["scaler"])
    logger.info("Scaler сохранён: %s", paths["scaler"])

    _atomic_json_dump(feature_cols, paths["features"])
    logger.info("Список признаков сохранён: %s", paths["features"])

    _atomic_json_dump(metrics, paths["metrics"])
    logger.info("Метрики сохранены: %s", paths["metrics"])


def verify_artifacts(config: Dict[str, Any]) -> None:
    """Проверяет, что сохранённые артефакты корректно загружаются и дают предсказания."""
    logger = logging.getLogger("churn_pipeline")
    artifacts_cfg = config["artifacts"]
    model_dir = Path(artifacts_cfg["model_dir"])

    model = joblib.load(model_dir / artifacts_cfg["model_name"])
    scaler = joblib.load(model_dir / artifacts_cfg["scaler_name"])
    with (model_dir / artifacts_cfg["features_name"]).open("r", encoding="utf-8") as f:
        feature_cols = json.load(f)

    df_sample = pd.read_parquet(config["data"]["features_path"]).head(100)
    X_sample = df_sample[feature_cols].replace([np.inf, -np.inf], 0).fillna(0)
    X_scaled = scaler.transform(X_sample)
    preds = model.predict(X_scaled)
    logger.info("Проверка инференса пройдена успешно. Получено предсказаний: %d", len(preds))


def run_pipeline(config_path: str, verify: bool = False) -> Tuple[RandomForestClassifier, Dict[str, float]]:
    """Оркестрирует полный цикл обучения и сохранения артефактов."""
    config = load_config(config_path)
    logger = setup_logging(config.get("logging", {}))
    logger.info("=== Старт конвейера (config=%s) ===", config_path)

    data_cfg = config["data"]
    df_model = load_data(
        features_path=data_cfg["features_path"],
        churn_path=data_cfg["churn_path"],
        id_col=data_cfg.get("id_col", "subscriber_key"),
        target_col=data_cfg.get("target_col", "churned"),
    )

    X_train, X_val, X_test, y_train, y_val, y_test, feature_cols, scaler = preprocess(
        df_model=df_model,
        preprocessing_cfg=config["preprocessing"],
        id_col=data_cfg.get("id_col", "subscriber_key"),
        target_col=data_cfg.get("target_col", "churned"),
    )

    model, train_time = train_model(X_train, y_train, config["model"])
    metrics = evaluate(model, X_val, y_val, X_test, y_test)
    metrics["train_time_sec"] = round(float(train_time), 4)

    save_artifacts(model, scaler, feature_cols, metrics, config["artifacts"])

    if verify:
        verify_artifacts(config)

    logger.info("=== Конвейер успешно завершён ===")
    return model, metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Churn Prediction ML Pipeline")
    parser.add_argument(
        "--config",
        type=str,
        default="config.json",
        help="Путь к JSON-файлу конфигурации (по умолчанию: config.json)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Выполнить проверку загрузки модели и тестового инференса после обучения",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(config_path=args.config, verify=args.verify)
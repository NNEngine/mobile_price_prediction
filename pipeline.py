import argparse
import logging
import yaml
from pathlib import Path

from src.data.load_data import load_processed_data
from src.data.reg_vs_clf import task_type
from src.models.select_model import select_model, model_list
from src.models.train_model import train_model
from src.models.predict_model import predict_model
from src.models.evaluate_model import evaluate_model

ROOT = Path(__file__).resolve().parent

with open(ROOT / "params.yaml") as f:
    params = yaml.safe_load(f)

logging.basicConfig(
    level=logging.INFO,
    format=params["logging"]["format"]
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# STAGE 1: Load Data
# ─────────────────────────────────────────────
def stage_load_data():
    logger.info("STAGE 1 — Loading data")
    df = load_processed_data()
    logger.info(f"Data loaded. Shape: {df.shape}")
    return df


# ─────────────────────────────────────────────
# STAGE 2: Detect Task Type
# ─────────────────────────────────────────────
def stage_task_type(df):
    logger.info("STAGE 2 — Detecting task type")
    result = task_type(df[params["target"]["column"]], "processed")
    logger.info(f"Task type: {result['type']}")
    return result["type"]


# ─────────────────────────────────────────────
# STAGE 3: Select Model
# ─────────────────────────────────────────────
def stage_select_model(task):
    logger.info("STAGE 3 — Selecting model")
    models = model_list(task)
    logger.info(f"Available models: {models}")
    selected = select_model(task)
    logger.info(f"Selected model: {selected}")
    return selected


# ─────────────────────────────────────────────
# STAGE 4: Feature Engineering (placeholder)
# ─────────────────────────────────────────────
def stage_features(df):
    """
    Placeholder for src/features/ pipeline.
    Will handle encoding, scaling, and train/test split.
    Replace this with actual feature engineering once src/features/ is built.
    """
    logger.info("STAGE 4 — Feature engineering (placeholder)")
    from sklearn.model_selection import train_test_split

    TARGET = params["target"]["column"]
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    logger.info(f"Train size: {X_train.shape}, Test size: {X_test.shape}")
    return X_train, X_test, y_train, y_test


# ─────────────────────────────────────────────
# STAGE 5: Train Model
# ─────────────────────────────────────────────
def stage_train(X_train, y_train):
    logger.info("STAGE 5 — Training model")
    model = train_model(X_train, y_train)
    logger.info("Training complete.")
    return model


# ─────────────────────────────────────────────
# STAGE 6: Predict
# ─────────────────────────────────────────────
def stage_predict(X_test, y_test):
    logger.info("STAGE 6 — Running predictions")
    comparison = predict_model(X_test, y_test)
    logger.info(f"Predictions complete. Shape: {comparison.shape}")
    return comparison


# ─────────────────────────────────────────────
# STAGE 7: Evaluate
# ─────────────────────────────────────────────
def stage_evaluate(comparison):
    logger.info("STAGE 7 — Evaluating model")
    report = evaluate_model(comparison)
    logger.info(f"Evaluation complete. Report: {report}")
    return report


# ─────────────────────────────────────────────
# FULL PIPELINE
# ─────────────────────────────────────────────
def run_pipeline():
    logger.info("=" * 50)
    logger.info("STARTING PIPELINE")
    logger.info("=" * 50)

    try:
        df              = stage_load_data()
        task            = stage_task_type(df)
        _               = stage_select_model(task)
        X_train, X_test, y_train, y_test = stage_features(df)
        _               = stage_train(X_train, y_train)
        comparison      = stage_predict(X_test, y_test)
        report          = stage_evaluate(comparison)

        logger.info("=" * 50)
        logger.info("PIPELINE COMPLETE")
        logger.info("=" * 50)
        return report

    except Exception as e:
        logger.error(f"Pipeline failed at: {e}")
        raise


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Mobile Price Prediction Pipeline"
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Run the full pipeline end to end."
    )
    parser.add_argument(
        "--stage",
        type=str,
        choices=["load", "task", "select", "features", "train", "predict", "evaluate"],
        help="Run a specific pipeline stage only."
    )
    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    if args.run:
        run_pipeline()

    elif args.stage:
        df = load_processed_data()
        task = params["selected_task"]

        stage_map = {
            "load":     lambda: stage_load_data(),
            "task":     lambda: stage_task_type(df),
            "select":   lambda: stage_select_model(task),
            "features": lambda: stage_features(df),
            "train":    lambda: stage_train(*stage_features(df)[:2]),
            "predict":  lambda: stage_predict(*stage_features(df)[2:]),
            "evaluate": lambda: stage_evaluate(
                            stage_predict(*stage_features(df)[2:])
                        ),
        }
        stage_map[args.stage]()

    else:
        parser.print_help()

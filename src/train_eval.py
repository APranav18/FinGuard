import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from .config import METADATA_PATH, MODEL_PATH, RANDOM_STATE


def get_model_candidates() -> Dict[str, object]:
    """Return candidate models for churn classification."""
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            min_samples_split=6,
            random_state=RANDOM_STATE,
        ),
    }


def evaluate_models(
    preprocessor,
    models: Dict[str, object],
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> Tuple[pd.DataFrame, Dict[str, Pipeline], Dict[str, np.ndarray]]:
    """Train all models and return metrics, fitted pipelines, and probability scores."""
    metrics_rows: List[Dict[str, float]] = []
    fitted_models: Dict[str, Pipeline] = {}
    probabilities: Dict[str, np.ndarray] = {}

    for model_name, model in models.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", clone(preprocessor)),
                ("classifier", model),
            ]
        )
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1] if hasattr(pipeline, "predict_proba") else None

        metrics_rows.append(
            {
                "Model": model_name,
                "Accuracy": accuracy_score(y_test, y_pred),
                "Precision": precision_score(y_test, y_pred, zero_division=0),
                "Recall": recall_score(y_test, y_pred, zero_division=0),
                "F1-Score": f1_score(y_test, y_pred, zero_division=0),
                "ROC-AUC": roc_auc_score(y_test, y_prob) if y_prob is not None else float("nan"),
            }
        )
        fitted_models[model_name] = pipeline
        if y_prob is not None:
            probabilities[model_name] = y_prob

    metrics_df = pd.DataFrame(metrics_rows).sort_values("F1-Score", ascending=False).reset_index(drop=True)
    return metrics_df, fitted_models, probabilities


def cross_validate_pipelines(
    pipelines: Dict[str, Pipeline],
    X: pd.DataFrame,
    y: pd.Series,
    cv_splits: int = 5,
) -> pd.DataFrame:
    """Run stratified cross-validation for pipelines and return mean/std summary."""
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=RANDOM_STATE)
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
    }

    rows: List[Dict[str, float]] = []
    for model_name, pipeline in pipelines.items():
        scores = cross_validate(
            estimator=clone(pipeline),
            X=X,
            y=y,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
        )

        row: Dict[str, float] = {"Model": model_name}
        for metric_key in scoring.keys():
            values = scores[f"test_{metric_key}"]
            row[f"{metric_key}_mean"] = float(np.mean(values))
            row[f"{metric_key}_std"] = float(np.std(values))
        rows.append(row)

    return pd.DataFrame(rows).sort_values("f1_mean", ascending=False).reset_index(drop=True)


def tune_models(
    preprocessor,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    score_metric: str = "f1",
) -> Dict[str, Pipeline]:
    """Tune candidate models using GridSearchCV and return fitted best pipelines."""
    search_space = {
        "Logistic Regression": (
            LogisticRegression(max_iter=4000, random_state=RANDOM_STATE),
            {
                "classifier__C": [0.2, 0.5, 1.0, 2.0],
                "classifier__class_weight": [None, "balanced"],
            },
        ),
        "Decision Tree": (
            DecisionTreeClassifier(random_state=RANDOM_STATE),
            {
                "classifier__max_depth": [4, 6, 8, 12],
                "classifier__min_samples_split": [2, 6, 12],
                "classifier__class_weight": [None, "balanced"],
            },
        ),
        "Random Forest": (
            RandomForestClassifier(random_state=RANDOM_STATE),
            {
                "classifier__n_estimators": [200, 400],
                "classifier__max_depth": [8, 12, None],
                "classifier__min_samples_split": [2, 6],
                "classifier__class_weight": [None, "balanced"],
            },
        ),
    }

    tuned_models: Dict[str, Pipeline] = {}
    for model_name, (estimator, params) in search_space.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", clone(preprocessor)),
                ("classifier", estimator),
            ]
        )
        grid = GridSearchCV(
            estimator=pipeline,
            param_grid=params,
            scoring=score_metric,
            cv=5,
            n_jobs=-1,
            verbose=0,
        )
        grid.fit(X_train, y_train)
        tuned_models[model_name] = grid.best_estimator_

    return tuned_models


def find_best_threshold(y_true: pd.Series, y_prob: np.ndarray) -> Dict[str, float]:
    """Find the threshold that maximizes F1 score."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)

    # precision_recall_curve returns an extra precision/recall point without threshold.
    precision = precision[:-1]
    recall = recall[:-1]

    f1_values = (2 * precision * recall) / (precision + recall + 1e-10)
    best_idx = int(np.argmax(f1_values))

    best_threshold = float(thresholds[best_idx])
    best_f1 = float(f1_values[best_idx])
    best_precision = float(precision[best_idx])
    best_recall = float(recall[best_idx])

    return {
        "threshold": best_threshold,
        "f1": best_f1,
        "precision": best_precision,
        "recall": best_recall,
    }


def evaluate_with_threshold(y_true: pd.Series, y_prob: np.ndarray, threshold: float) -> Dict[str, float]:
    """Evaluate classification metrics at a custom probability threshold."""
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "F1-Score": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def plot_confusion_matrix(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_path: Path,
    title: str,
    threshold: float | None = None,
) -> None:
    """Save confusion matrix visualization for a selected model."""
    if threshold is None:
        y_pred = model.predict(X_test)
    else:
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= threshold).astype(int)

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Not Churn", "Churn"])

    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_roc_curves(y_test: pd.Series, probabilities: Dict[str, np.ndarray], output_path: Path) -> None:
    """Save ROC curve plot for all probabilistic models."""
    plt.figure(figsize=(8, 6))
    for model_name, y_prob in probabilities.items():
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc_score = roc_auc_score(y_test, y_prob)
        plt.plot(fpr, tpr, linewidth=2, label=f"{model_name} (AUC={auc_score:.3f})")

    plt.plot([0, 1], [0, 1], "k--", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_feature_importance(
    random_forest_model: Pipeline,
    output_path: Path,
    top_n: int = 12,
) -> None:
    """Save feature importance chart for the trained random forest model."""
    preprocessor = random_forest_model.named_steps["preprocessor"]
    classifier = random_forest_model.named_steps["classifier"]

    feature_names = preprocessor.get_feature_names_out()
    importances = classifier.feature_importances_

    importance_df = (
        pd.DataFrame({"Feature": feature_names, "Importance": importances})
        .sort_values("Importance", ascending=False)
        .head(top_n)
    )

    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=importance_df,
        x="Importance",
        y="Feature",
        hue="Feature",
        palette="viridis",
        dodge=False,
        legend=False,
    )
    plt.title("Top Feature Importances (Random Forest)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_best_model(best_model: Pipeline, best_model_name: str, feature_columns: List[str], category_map: Dict[str, List[str]]) -> None:
    """Persist best model and metadata."""
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(best_model, f)

    metadata = {
        "best_model_name": best_model_name,
        "feature_columns": feature_columns,
        "categorical_options": category_map,
        "model_path": str(MODEL_PATH),
        "prediction_threshold": 0.5,
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def save_best_model_with_metadata(
    best_model: Pipeline,
    best_model_name: str,
    feature_columns: List[str],
    category_map: Dict[str, List[str]],
    prediction_threshold: float,
    threshold_metrics: Dict[str, float],
    is_tuned: bool,
) -> None:
    """Persist best model and richer metadata used by the app and reports."""
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(best_model, f)

    metadata = {
        "best_model_name": best_model_name,
        "feature_columns": feature_columns,
        "categorical_options": category_map,
        "model_path": str(MODEL_PATH),
        "prediction_threshold": float(prediction_threshold),
        "threshold_metrics": threshold_metrics,
        "is_tuned": is_tuned,
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

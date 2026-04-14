import argparse
from pathlib import Path

from src.config import DATA_PATH, FIGURES_DIR, REPORTS_DIR
from src.eda import generate_eda
from src.preprocessing import build_preprocessor, load_dataset, prepare_features_and_target, split_data
from src.reporting import write_markdown_summary_report
from src.train_eval import (
    cross_validate_pipelines,
    evaluate_models,
    evaluate_with_threshold,
    find_best_threshold,
    get_model_candidates,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_roc_curves,
    save_best_model_with_metadata,
    tune_models,
)


def run_pipeline(dataset_path: str, tune: bool, optimize_threshold: bool) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset from: {dataset_path}")
    df = load_dataset(dataset_path)

    print("Running EDA and saving visualizations...")
    generate_eda(
        df=df,
        figures_dir=FIGURES_DIR,
        insights_path=REPORTS_DIR / "eda_insights.txt",
    )

    print("Preparing data for training...")
    X, y, numeric_features, categorical_features = prepare_features_and_target(df)
    preprocessor = build_preprocessor(numeric_features, categorical_features)
    X_train, X_test, y_train, y_test = split_data(X, y)

    print("Training and evaluating candidate models...")
    if tune:
        print("Hyperparameter tuning enabled (GridSearchCV).")
        fitted_models = tune_models(preprocessor=preprocessor, X_train=X_train, y_train=y_train)
        metrics_df, fitted_models, probabilities = evaluate_models(
            preprocessor=preprocessor,
            models={name: model.named_steps["classifier"] for name, model in fitted_models.items()},
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
        )
    else:
        models = get_model_candidates()
        metrics_df, fitted_models, probabilities = evaluate_models(
            preprocessor=preprocessor,
            models=models,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
        )

    metrics_path = REPORTS_DIR / "model_comparison.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Saved model comparison: {metrics_path}")
    print(metrics_df.to_string(index=False))

    cv_report_df = cross_validate_pipelines(pipelines=fitted_models, X=X_train, y=y_train)
    cv_report_path = REPORTS_DIR / "cross_validation_report.csv"
    cv_report_df.to_csv(cv_report_path, index=False)
    print(f"Saved cross-validation report: {cv_report_path}")

    best_model_name = metrics_df.iloc[0]["Model"]
    best_model = fitted_models[best_model_name]

    best_threshold = 0.5
    threshold_metrics = {}
    if optimize_threshold:
        best_prob = best_model.predict_proba(X_test)[:, 1]
        threshold_result = find_best_threshold(y_true=y_test, y_prob=best_prob)
        best_threshold = threshold_result["threshold"]
        threshold_metrics = evaluate_with_threshold(y_true=y_test, y_prob=best_prob, threshold=best_threshold)

        threshold_report_path = REPORTS_DIR / "threshold_optimization.txt"
        threshold_lines = [
            f"Best threshold (max F1): {best_threshold:.4f}",
            f"Precision @ threshold: {threshold_metrics['Precision']:.4f}",
            f"Recall @ threshold: {threshold_metrics['Recall']:.4f}",
            f"F1-Score @ threshold: {threshold_metrics['F1-Score']:.4f}",
            f"Accuracy @ threshold: {threshold_metrics['Accuracy']:.4f}",
        ]
        threshold_report_path.write_text("\n".join(threshold_lines), encoding="utf-8")
        print(f"Saved threshold optimization report: {threshold_report_path}")

    plot_confusion_matrix(
        model=best_model,
        X_test=X_test,
        y_test=y_test,
        output_path=FIGURES_DIR / "best_model_confusion_matrix.png",
        title=f"Confusion Matrix - {best_model_name}",
        threshold=best_threshold if optimize_threshold else None,
    )
    print("Saved confusion matrix plot.")

    if probabilities:
        plot_roc_curves(y_test=y_test, probabilities=probabilities, output_path=FIGURES_DIR / "roc_curve.png")
        print("Saved ROC curve plot.")

    if "Random Forest" in fitted_models:
        plot_feature_importance(
            random_forest_model=fitted_models["Random Forest"],
            output_path=FIGURES_DIR / "random_forest_feature_importance.png",
        )
        print("Saved random forest feature importance plot.")

    category_map = {col: sorted(df[col].dropna().astype(str).unique().tolist()) for col in categorical_features}
    save_best_model_with_metadata(
        best_model=best_model,
        best_model_name=best_model_name,
        feature_columns=X.columns.tolist(),
        category_map=category_map,
        prediction_threshold=best_threshold,
        threshold_metrics=threshold_metrics,
        is_tuned=tune,
    )

    summary_report_path = REPORTS_DIR / "training_summary.md"
    write_markdown_summary_report(
        output_path=summary_report_path,
        metrics_df=metrics_df,
        cv_report_df=cv_report_df,
        best_model_name=best_model_name,
        is_tuned=tune,
        threshold=best_threshold,
        threshold_metrics=threshold_metrics,
    )
    print(f"Saved markdown summary report: {summary_report_path}")
    print("Best model saved to models/best_model.pkl")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Customer Churn Prediction Training Pipeline")
    parser.add_argument(
        "--data",
        type=str,
        default=str(DATA_PATH),
        help="Path to the Kaggle churn CSV file.",
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Enable hyperparameter tuning using GridSearchCV.",
    )
    parser.add_argument(
        "--optimize-threshold",
        action="store_true",
        help="Find a probability threshold that maximizes F1-score on test set.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    dataset_file = Path(args.data)

    if not dataset_file.exists():
        raise FileNotFoundError(
            f"Dataset not found at '{dataset_file}'. Place the Kaggle CSV at data/Churn_Modelling.csv or pass --data."
        )

    run_pipeline(
        str(dataset_file),
        tune=args.tune,
        optimize_threshold=args.optimize_threshold,
    )

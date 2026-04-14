from pathlib import Path
from typing import Dict

import pandas as pd


def _format_metric_table(df: pd.DataFrame) -> str:
    """Render a dataframe as a fixed-width markdown code block."""
    if df.empty:
        return "No data available."
    return "```text\n" + df.to_string(index=False) + "\n```"


def write_markdown_summary_report(
    output_path: Path,
    metrics_df: pd.DataFrame,
    cv_report_df: pd.DataFrame,
    best_model_name: str,
    is_tuned: bool,
    threshold: float,
    threshold_metrics: Dict[str, float],
) -> None:
    """Create a single markdown report combining core training outputs."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Training Summary Report",
        "",
        "## Run Configuration",
        f"- Best model: {best_model_name}",
        f"- Hyperparameter tuning: {'Enabled' if is_tuned else 'Disabled'}",
        f"- Prediction threshold: {threshold:.4f}",
        "",
        "## Test Set Model Comparison",
        _format_metric_table(metrics_df.round(4)),
        "",
        "## Cross-Validation Summary (Train Split)",
        _format_metric_table(cv_report_df.round(4)),
        "",
        "## Threshold Metrics",
    ]

    if threshold_metrics:
        lines.extend(
            [
                f"- Accuracy: {threshold_metrics.get('Accuracy', 0.0):.4f}",
                f"- Precision: {threshold_metrics.get('Precision', 0.0):.4f}",
                f"- Recall: {threshold_metrics.get('Recall', 0.0):.4f}",
                f"- F1-Score: {threshold_metrics.get('F1-Score', 0.0):.4f}",
            ]
        )
    else:
        lines.append("- Threshold optimization not enabled; default threshold 0.5000 used.")

    output_path.write_text("\n".join(lines), encoding="utf-8")

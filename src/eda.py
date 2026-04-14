from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .config import TARGET_COLUMN


def generate_eda(df: pd.DataFrame, figures_dir: Path, insights_path: Path) -> Dict[str, str]:
    """Generate core EDA visuals and save concise business insights."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    insights_path.parent.mkdir(parents=True, exist_ok=True)

    saved_files = {}

    plt.figure(figsize=(7, 5))
    ax = sns.countplot(
        data=df,
        x=TARGET_COLUMN,
        hue=TARGET_COLUMN,
        palette="Set2",
        legend=False,
    )
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Not Churn", "Churn"])
    ax.set_title("Distribution of Churn vs Non-Churn")
    ax.set_xlabel("Customer Status")
    ax.set_ylabel("Count")
    churn_plot = figures_dir / "churn_distribution.png"
    plt.tight_layout()
    plt.savefig(churn_plot, dpi=200)
    plt.close()
    saved_files["churn_distribution"] = str(churn_plot)

    numeric_df = df.select_dtypes(include=["number"]).copy()
    if TARGET_COLUMN in df.columns and TARGET_COLUMN not in numeric_df.columns:
        numeric_df[TARGET_COLUMN] = df[TARGET_COLUMN]

    plt.figure(figsize=(12, 8))
    corr = numeric_df.corr(numeric_only=True)
    sns.heatmap(corr, cmap="coolwarm", center=0, annot=False)
    plt.title("Feature Correlation Heatmap")
    corr_plot = figures_dir / "correlation_heatmap.png"
    plt.tight_layout()
    plt.savefig(corr_plot, dpi=200)
    plt.close()
    saved_files["correlation_heatmap"] = str(corr_plot)

    churn_rate = df[TARGET_COLUMN].mean() * 100
    geography_insight = (
        df.groupby("Geography", as_index=False)[TARGET_COLUMN]
        .mean()
        .sort_values(TARGET_COLUMN, ascending=False)
    )
    gender_insight = (
        df.groupby("Gender", as_index=False)[TARGET_COLUMN]
        .mean()
        .sort_values(TARGET_COLUMN, ascending=False)
    )

    top_corr = corr[TARGET_COLUMN].drop(TARGET_COLUMN).abs().sort_values(ascending=False).head(5)

    insights_lines = [
        "Key EDA Insights",
        "================",
        f"Overall churn rate: {churn_rate:.2f}%",
        "",
        "Churn rate by geography:",
        geography_insight.to_string(index=False),
        "",
        "Churn rate by gender:",
        gender_insight.to_string(index=False),
        "",
        "Top absolute correlations with churn:",
        top_corr.to_string(),
    ]

    insights_path.write_text("\n".join(insights_lines), encoding="utf-8")
    saved_files["insights"] = str(insights_path)

    return saved_files

from typing import List, Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import CATEGORICAL_FEATURES, DROP_COLUMNS, RANDOM_STATE, TARGET_COLUMN, TEST_SIZE


def load_dataset(csv_path: str) -> pd.DataFrame:
    """Load the churn dataset from a CSV file."""
    return pd.read_csv(csv_path)


def prepare_features_and_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str], List[str]]:
    """Drop non-informative columns and return X, y with validated feature groups."""
    working_df = df.copy()

    if TARGET_COLUMN not in working_df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found in dataset.")

    for col in DROP_COLUMNS:
        if col in working_df.columns:
            working_df = working_df.drop(columns=col)

    y = working_df[TARGET_COLUMN]
    X = working_df.drop(columns=[TARGET_COLUMN])

    categorical_features = [col for col in CATEGORICAL_FEATURES if col in X.columns]
    numeric_features = [col for col in X.columns if col not in categorical_features]

    return X, y, numeric_features, categorical_features


def build_preprocessor(numeric_features: List[str], categorical_features: List[str]) -> ColumnTransformer:
    """Create a preprocessing transformer for numeric and categorical columns."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )

    return preprocessor


def split_data(X: pd.DataFrame, y: pd.Series):
    """Split features and target into train and test sets."""
    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

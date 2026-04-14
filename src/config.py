from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "Churn_Modelling.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "best_model.pkl"
METADATA_PATH = PROJECT_ROOT / "models" / "model_metadata.json"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

TARGET_COLUMN = "Exited"
DROP_COLUMNS = ["RowNumber", "CustomerId", "Surname"]

RANDOM_STATE = 42
TEST_SIZE = 0.2

CATEGORICAL_FEATURES = ["Geography", "Gender"]
NUMERIC_FEATURES = [
    "CreditScore",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
    "EstimatedSalary",
]

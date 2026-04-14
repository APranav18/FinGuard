# Customer Churn Prediction in Banking Using Machine Learning

End-to-end machine learning project for predicting customer churn in a banking dataset (Kaggle CSV format), including EDA, preprocessing, model comparison, evaluation, model persistence, and a Streamlit UI.

## Project Structure

```text
hackthonbank/
├── app/
│   └── streamlit_app.py
├── .streamlit/
│   └── config.toml
├── data/
│   └── Churn_Modelling.csv   # add Kaggle dataset here
├── Dockerfile
├── .dockerignore
├── models/
│   ├── best_model.pkl        # generated after training
│   └── model_metadata.json   # generated after training
├── notebooks/
│   └── churn_eda.ipynb
├── reports/
│   ├── eda_insights.txt
│   ├── model_comparison.csv
│   ├── cross_validation_report.csv
│   ├── training_summary.md
│   └── figures/
│       ├── churn_distribution.png
│       ├── correlation_heatmap.png
│       ├── best_model_confusion_matrix.png
│       ├── random_forest_feature_importance.png
│       └── roc_curve.png
├── src/
│   ├── config.py
│   ├── eda.py
│   ├── preprocessing.py
│   └── train_eval.py
├── main.py
└── requirements.txt
```

## Dataset

Use the Kaggle **Bank Customer Churn** dataset (CSV). Place it at:

- `data/Churn_Modelling.csv`

Expected target column:

- `Exited` (1 = churn, 0 = not churn)

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Train + Evaluate

```bash
python main.py --data data/Churn_Modelling.csv
```

Advanced run with hyperparameter tuning and threshold optimization:

```bash
python main.py --data data/Churn_Modelling.csv --tune --optimize-threshold
```

This command performs:

- Missing value handling with imputers
- Categorical encoding (`Gender`, `Geography`) via one-hot encoding
- Numerical scaling via standardization
- Train/test split with stratification
- Model training and comparison:
  - Logistic Regression
  - Decision Tree
  - Random Forest
- Metrics:
  - Accuracy
  - Precision
  - Recall
  - F1-score
  - ROC-AUC
  - Confusion Matrix plot (best model)
- Additional plots:
  - Churn distribution
  - Correlation heatmap
  - Feature importance (Random Forest)
  - ROC curve
- Best model selection by F1-score
- Cross-validation summary report (mean/std across folds)
- Consolidated markdown summary report (`reports/training_summary.md`)
- Optional hyperparameter tuning using GridSearchCV (`--tune`)
- Optional threshold optimization for churn probability (`--optimize-threshold`)
- Model persistence using pickle (`models/best_model.pkl`)

## Run Streamlit App

```bash
streamlit run app/streamlit_app.py
```

The UI allows input for customer features such as age, balance, credit score, tenure, products, gender, geography, and more. It predicts:

- `CHURN` or `NOT CHURN`
- Churn probability
- Uses saved prediction threshold from training metadata
- Slider-based inputs for interactive demos
- Real-time prediction updates as inputs change
- Built-in charts:
  - Probability chart (Churn vs Stay)
  - Input vs dataset-average feature comparison
- Business insight panel:
  - Likely customer demerits (pain points)
  - Retention actions the bank can take to reduce churn
- Interaction-based prediction enhancement:
  - Inputs for online engagement and usage frequency
  - Behavior-adjusted churn probability (base ML score + digital interaction signals)
  - Configurable behavior weighting mode: Conservative, Balanced, Aggressive
  - Dedicated section for interaction frequency and feature research frequency
- Customer lookup autofill:
  - Search by account number (CustomerId) or name (Surname)
  - Auto-fills profile inputs and inferred behavior for instant prediction

## Deployment (Docker)

Build image:

```bash
docker build -t bank-churn-app .
```

Run container:

```bash
docker run -p 8501:8501 bank-churn-app
```

## Notes

- Keep feature names aligned with the Kaggle CSV schema.
- If your dataset path is different, pass it using `--data`.

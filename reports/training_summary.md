# Training Summary Report

## Run Configuration
- Best model: Random Forest
- Hyperparameter tuning: Enabled
- Prediction threshold: 0.4663

## Test Set Model Comparison
```text
              Model  Accuracy  Precision  Recall  F1-Score  ROC-AUC
      Random Forest    0.8480     0.6388  0.5823    0.6093   0.8595
      Decision Tree    0.7595     0.4474  0.7740    0.5671   0.8397
Logistic Regression    0.7135     0.3872  0.7002    0.4987   0.7772
```

## Cross-Validation Summary (Train Split)
```text
              Model  accuracy_mean  accuracy_std  precision_mean  precision_std  recall_mean  recall_std  f1_mean  f1_std  roc_auc_mean  roc_auc_std
      Random Forest         0.8472        0.0094          0.6365         0.0211       0.5816      0.0405   0.6075  0.0307        0.8560       0.0119
      Decision Tree         0.7565        0.0061          0.4437         0.0083       0.7681      0.0369   0.5622  0.0138        0.8280       0.0150
Logistic Regression         0.7078        0.0114          0.3798         0.0143       0.6859      0.0358   0.4887  0.0190        0.7662       0.0205
```

## Threshold Metrics
- Accuracy: 0.8450
- Precision: 0.6174
- Recall: 0.6265
- F1-Score: 0.6220
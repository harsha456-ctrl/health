# Project Report — Explainable, IoT-Deployable Early-Warning System for Heart Failure Mortality Risk

**Course:** Predictive Models in IoT Using ML  
**Instructor:** Dr. N. Vikram  
**Student:** A V Harshavardhanreddy  
**USN:** 23BTRCO002  
**Dataset:** UCI Heart Failure Clinical Records (ID #519)  
**Date:** September 27, 2026

## 1. Problem Statement

This project studies machine-learning prediction of mortality risk among patients with heart failure using the UCI Heart Failure Clinical Records dataset. The project report frames the system as an explainable and IoT-deployable early-warning system.

## 2. Dataset

The selected dataset is the UCI Heart Failure Clinical Records dataset (ID #519). The target variable is **DEATH_EVENT**.

The implementation in `src/train.py` downloads the dataset, performs a stratified 80/20 train-test split, and evaluates multiple classifiers. It also performs 5-fold stratified cross-validation.

## 3. Models Implemented

- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting
- SVM with RBF kernel
- Extra Trees

Evaluation metrics:
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Matthews Correlation Coefficient (MCC)

## 4. Results Reported in the Project Report

The submitted project report reports the following experimental results:

| Study / Model | Accuracy | MCC | F1 |
|---|---:|---:|---:|
| Chicco & Jurman (2020) — Random Forest | 0.8350 | 0.4000 | 0.6500 |
| Ishaq et al. (2021) — SMOTE + Extra Trees | 0.9260 | 0.8400 | 0.9200 |
| Umer et al. (2022) — CNN/MLP/RNN/LSTM | 0.9280 | 0.8200 | 0.9100 |
| Our Work — LightGBM, 5-Fold CV | 0.8494 | 0.6486 | 0.7551 |
| Our Work — Extra Trees + SMOTE + Edge Tuning | 0.8361 | 0.6283 | 0.7440 |

The report also describes SHAP-based explainability, streaming/IoT operation, and edge inference latency.

**Important reproducibility note:** the table above records the results stated in the supplied project report. The GitHub training script is a separate reproducible baseline implementation and does not claim to reproduce these exact LightGBM/SMOTE/SHAP numbers.

## 5. Comparison With Published Work

The report compares the proposed system with published approaches on the same dataset. The comparison considers predictive metrics as well as deployment-oriented characteristics such as streaming operation, explainability, and edge deployment.

## 6. Reproducibility

Run:

```bash
pip install -r requirements.txt
python src/train.py
```

The script saves:
- `results/holdout_results.csv`
- `results/5fold_cv_results.csv`
- `results/dataset_summary.csv`
- `results/roc_curves.png`
- classification reports
- fitted models in `models/`

## 7. Conclusion

The project demonstrates a machine-learning workflow for heart-failure mortality-risk prediction and discusses how explainability and edge deployment can be incorporated into an IoT-oriented healthcare system. The repository contains the source implementation and documentation needed to reproduce the baseline experiment.

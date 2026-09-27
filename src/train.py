"""
Heart Failure Clinical Records - ML Training Pipeline
UCI Dataset 519.

Downloads the dataset, trains baseline classifiers, evaluates a stratified
20% hold-out set, performs 5-fold stratified cross-validation, and saves
metrics, ROC curves, reports, and fitted models.

The numbers produced by this script are this implementation's results; they
should not be presented as exact reproductions of a published paper unless
the experimental protocol is independently matched.
"""
from pathlib import Path
import urllib.request, zipfile
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, matthews_corrcoef, confusion_matrix,
                             classification_report, RocCurveDisplay)

RANDOM_STATE = 42
DATA_URL = "https://archive.ics.uci.edu/static/public/519/heart%2Bfailure%2Bclinical%2Brecords.zip"
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR, RESULTS_DIR, MODELS_DIR = ROOT/"data", ROOT/"results", ROOT/"models"
for d in (DATA_DIR, RESULTS_DIR, MODELS_DIR):
    d.mkdir(exist_ok=True)

def load_dataset():
    csv_path = DATA_DIR/"heart_failure_clinical_records_dataset.csv"
    zip_path = DATA_DIR/"heart_failure.zip"
    if not csv_path.exists():
        print("Downloading UCI dataset...")
        urllib.request.urlretrieve(DATA_URL, zip_path)
        with zipfile.ZipFile(zip_path) as z:
            names = [n for n in z.namelist() if n.lower().endswith(".csv")]
            if not names:
                raise FileNotFoundError("No CSV found in UCI archive.")
            z.extract(names[0], DATA_DIR)
            extracted = DATA_DIR/names[0]
            if extracted != csv_path:
                extracted.replace(csv_path)
    return pd.read_csv(csv_path)

def build_models():
    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE))
        ]),
        "Decision Tree": DecisionTreeClassifier(max_depth=4, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=500, min_samples_leaf=2, class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
        "SVM (RBF)": Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE))
        ]),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=500, class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=-1)
    }

def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    score = (model.predict_proba(X_test)[:, 1]
             if hasattr(model, "predict_proba")
             else model.decision_function(X_test))
    metrics = {
        "model": name,
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "f1": f1_score(y_test, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, score),
        "mcc": matthews_corrcoef(y_test, pred)
    }
    report = classification_report(y_test, pred, zero_division=0)
    print(f"\n{name}\n{'-'*len(name)}")
    print(pd.Series(metrics).to_string())
    print("\nConfusion matrix:\n", confusion_matrix(y_test, pred))
    print("\nClassification report:\n", report)
    safe = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    (RESULTS_DIR/f"{safe}_report.txt").write_text(report, encoding="utf-8")
    return model, metrics, score

def main():
    df = load_dataset()
    target = "DEATH_EVENT"
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found.")
    X, y = df.drop(columns=[target]), df[target].astype(int)
    print("Dataset shape:", df.shape)
    print("Target distribution:\n", y.value_counts().sort_index())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE)

    models = build_models()
    holdout_rows, fitted = [], {}
    plt.figure(figsize=(8, 6))
    for name, model in models.items():
        fitted_model, metrics, score = evaluate_model(
            name, model, X_train, X_test, y_train, y_test)
        fitted[name] = fitted_model
        holdout_rows.append(metrics)
        RocCurveDisplay.from_predictions(y_test, score, name=name)
    plt.title("ROC Curves - Heart Failure Mortality Prediction")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR/"roc_curves.png", dpi=200)
    plt.close()

    pd.DataFrame(holdout_rows).to_csv(RESULTS_DIR/"holdout_results.csv", index=False)

    scoring = {
        "accuracy": "accuracy", "precision": "precision", "recall": "recall",
        "f1": "f1", "roc_auc": "roc_auc", "mcc": "matthews_corrcoef"
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_rows = []
    for name, model in models.items():
        scores = cross_validate(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
        row = {"model": name}
        for metric in scoring:
            values = scores[f"test_{metric}"]
            row[f"{metric}_mean"] = values.mean()
            row[f"{metric}_std"] = values.std()
        cv_rows.append(row)
    pd.DataFrame(cv_rows).to_csv(RESULTS_DIR/"5fold_cv_results.csv", index=False)

    for name, model in fitted.items():
        safe = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        joblib.dump(model, MODELS_DIR/f"{safe}.joblib")

    pd.DataFrame({
        "property": ["rows","columns","target","positive_class_count",
                     "negative_class_count","train_rows","test_rows",
                     "random_state","cv_folds"],
        "value": [len(df),len(df.columns),target,int((y==1).sum()),int((y==0).sum()),
                  len(X_train),len(X_test),RANDOM_STATE,5]
    }).to_csv(RESULTS_DIR/"dataset_summary.csv", index=False)

    print("\nSaved results/, models/, and dataset summary.")

if __name__ == "__main__":
    main()

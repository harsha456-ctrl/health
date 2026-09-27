import numpy as np
import time
import sys
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, roc_auc_score, confusion_matrix
)

def evaluate_classifier(model, X_test, y_test, X_train=None, y_train=None):
    """
    Evaluates a trained classifier and returns a comprehensive metrics dictionary.
    Includes classification metrics, MCC (critical baseline metric), ROC-AUC,
    and IoT Edge operational metrics (latency & model size).
    """
    # Record inference time
    start_time = time.perf_counter()
    y_pred = model.predict(X_test)
    end_time = time.perf_counter()
    
    total_latency_ms = (end_time - start_time) * 1000.0
    avg_latency_per_sample_ms = total_latency_ms / len(X_test)
    
    # Predict probabilities if supported
    y_prob = None
    if hasattr(model, "predict_proba"):
        try:
            y_prob = model.predict_proba(X_test)[:, 1]
        except Exception:
            y_prob = None
            
    # Confusion matrix elements
    cm = confusion_matrix(y_test, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn = fp = fn = tp = 0

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0) # Sensitivity
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = f1_score(y_test, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_test, y_pred)
    
    auc = roc_auc_score(y_test, y_prob) if y_prob is not None else np.nan

    return {
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall_sensitivity": round(float(rec), 4),
        "specificity": round(float(spec), 4),
        "f1_score": round(float(f1), 4),
        "mcc": round(float(mcc), 4),
        "roc_auc": round(float(auc), 4) if not np.isnan(auc) else None,
        "latency_per_sample_ms": round(float(avg_latency_per_sample_ms), 5),
        "confusion_matrix": cm.tolist(),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn)
    }

def print_metrics_table(results):
    """
    Utility function to display metrics formatted cleanly in console.
    """
    header = f"{'Model':<30} | {'Acc':<7} | {'Prec':<7} | {'Rec/Sens':<8} | {'Spec':<7} | {'F1':<7} | {'MCC':<7} | {'AUC':<7} | {'Lat(ms)':<8}"
    print("-" * len(header))
    print(header)
    print("-" * len(header))
    for name, res in results.items():
        auc_str = f"{res['roc_auc']:.4f}" if res['roc_auc'] is not None else "N/A"
        print(f"{name:<30} | {res['accuracy']:<7.4f} | {res['precision']:<7.4f} | {res['recall_sensitivity']:<8.4f} | {res['specificity']:<7.4f} | {res['f1_score']:<7.4f} | {res['mcc']:<7.4f} | {auc_str:<7} | {res['latency_per_sample_ms']:<8.4f}")
    print("-" * len(header))

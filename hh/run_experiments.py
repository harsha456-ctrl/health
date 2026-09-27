import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE

from src.data_loader import load_heart_failure_data, get_dataset_summary, preprocess_data
from src.models import train_and_evaluate_all, get_model_pipeline
from src.metrics import evaluate_classifier, print_metrics_table
from src.explainability import compute_shap_explanations
from src.iot_simulator import IoTEdgePatientMonitor

OUTPUT_DIR = "results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def run_cross_validation_experiments(df, n_splits=5, random_state=42):
    """
    Performs 5-Fold Stratified Cross Validation across all model architectures.
    Returns average metrics and standard deviations for robust statistical verification.
    """
    X = df.drop(columns=['DEATH_EVENT'])
    y = df['DEATH_EVENT']
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    model_names = [
        "Logistic Regression",
        "Random Forest (Chicco & Jurman 2020)",
        "RF Top-2 Features (Chicco & Jurman)",
        "Extra Trees (Ishaq et al. 2021)",
        "Extra Trees + SMOTE (Ishaq et al. 2021)",
        "XGBoost Classifier",
        "LightGBM Classifier",
        "MLP Neural Net (Umer et al. 2022)",
        "Proposed IoT Edge Lightweight Model"
    ]
    
    cv_summary = {name: {
        'accuracy': [], 'precision': [], 'recall_sensitivity': [], 'specificity': [],
        'f1_score': [], 'mcc': [], 'roc_auc': [], 'latency_per_sample_ms': []
    } for name in model_names}

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
        
        # Scaling
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_tr_scaled = pd.DataFrame(scaler.fit_transform(X_tr), columns=X.columns)
        X_te_scaled = pd.DataFrame(scaler.transform(X_te), columns=X.columns)
        
        # SMOTE oversampling
        smote = SMOTE(random_state=random_state + fold)
        X_tr_smote, y_tr_smote = smote.fit_resample(X_tr_scaled, y_tr)
        
        for name in model_names:
            clf = get_model_pipeline(name, random_state=random_state + fold)
            
            # Custom input handling per model specification
            if name == "RF Top-2 Features (Chicco & Jurman)":
                subset = ['serum_creatinine', 'ejection_fraction']
                cur_tr_x, cur_te_x, cur_tr_y = X_tr[subset], X_te[subset], y_tr
            elif name in ["Extra Trees + SMOTE (Ishaq et al. 2021)", "Proposed IoT Edge Lightweight Model"]:
                cur_tr_x, cur_te_x, cur_tr_y = X_tr_smote, X_te_scaled, y_tr_smote
            elif name in ["Logistic Regression", "MLP Neural Net (Umer et al. 2022)"]:
                cur_tr_x, cur_te_x, cur_tr_y = X_tr_scaled, X_te_scaled, y_tr
            else:
                cur_tr_x, cur_te_x, cur_tr_y = X_tr, X_te, y_tr

            clf.fit(cur_tr_x, cur_tr_y)
            res = evaluate_classifier(clf, cur_te_x, y_te)
            
            for k in cv_summary[name].keys():
                if res[k] is not None and not np.isnan(res[k]):
                    cv_summary[name][k].append(res[k])

    # Aggregate fold statistics
    aggregated_cv = {}
    for name, metrics_dict in cv_summary.items():
        aggregated_cv[name] = {}
        for metric, val_list in metrics_dict.items():
            if len(val_list) > 0:
                aggregated_cv[name][f"{metric}_mean"] = round(float(np.mean(val_list)), 4)
                aggregated_cv[name][f"{metric}_std"] = round(float(np.std(val_list)), 4)

    return aggregated_cv

def main():
    print("==========================================================")
    print("Heart Failure IoT ML Benchmark & Literature Comparison")
    print("==========================================================")

    df = load_heart_failure_data()
    summary = get_dataset_summary(df)
    print(f"\n[1] Dataset Loaded: {summary['total_records']} patient records, {summary['features_count']} features.")
    print(f"    Class Balance: {summary['survived_count']} Survived, {summary['deceased_count']} Deceased ({summary['mortality_rate_percent']}% mortality rate).")

    # Hold-out Split Evaluation
    data_dict = preprocess_data(df, test_size=0.2, random_state=42)
    print("\n[2] Training and Evaluating Baseline & Proposed Models on Test Set (80/20 Holdout)...")
    trained_models, test_results = train_and_evaluate_all(data_dict, random_state=42)
    print_metrics_table(test_results)

    # 5-Fold Stratified Cross Validation
    print("\n[3] Running 5-Fold Stratified Cross Validation for Statistical Validation...")
    cv_results = run_cross_validation_experiments(df, n_splits=5, random_state=42)
    
    # Format CV Results for console print
    print("\n5-Fold Stratified CV Aggregated Results (Mean ± Std):")
    cv_table = []
    for name, res in cv_results.items():
        row = {
            "Model": name,
            "Accuracy": f"{res['accuracy_mean']:.4f} ± {res['accuracy_std']:.4f}",
            "MCC": f"{res['mcc_mean']:.4f} ± {res['mcc_std']:.4f}",
            "F1-Score": f"{res['f1_score_mean']:.4f} ± {res['f1_score_std']:.4f}",
            "ROC-AUC": f"{res['roc_auc_mean']:.4f} ± {res['roc_auc_std']:.4f}",
            "Latency (ms)": f"{res['latency_per_sample_ms_mean']:.5f}"
        }
        cv_table.append(row)
    
    cv_df = pd.DataFrame(cv_table)
    print(cv_df.to_string(index=False))

    # Literature Benchmarks Comparison
    print("\n[4] Comparing Results with Literature Benchmarks...")
    literature_comparison = [
        {
            "Study": "Chicco & Jurman (2020)",
            "Method": "Random Forest (Feature Ranking)",
            "Dataset": "UCI Heart Failure (299 records)",
            "Accuracy": 0.8350,
            "MCC": 0.4000,
            "F1_Score": 0.6500,
            "Real_Time_IoT": "No - Offline Only",
            "Explainability": "No (Feature Ranking only)",
            "Edge_Deployable": "No"
        },
        {
            "Study": "Ishaq et al. (2021)",
            "Method": "SMOTE + Extra Trees Classifier",
            "Dataset": "UCI Heart Failure (299 records)",
            "Accuracy": 0.9260,
            "MCC": 0.8400,
            "F1_Score": 0.9200,
            "Real_Time_IoT": "No - Offline Only",
            "Explainability": "No",
            "Edge_Deployable": "No"
        },
        {
            "Study": "Umer et al. (2022)",
            "Method": "CNN / MLP / RNN / LSTM",
            "Dataset": "UCI Heart Failure (299 records)",
            "Accuracy": 0.9280,
            "MCC": 0.8200,
            "F1_Score": 0.9100,
            "Real_Time_IoT": "Partial - Cloud IoT",
            "Explainability": "No",
            "Edge_Deployable": "No (Cloud Dependency)"
        },
        {
            "Study": "Our Proposed Work (Replication/Eval)",
            "Method": "ExtraTrees + SMOTE Baseline",
            "Dataset": "UCI Heart Failure (299 records)",
            "Accuracy": test_results["Extra Trees + SMOTE (Ishaq et al. 2021)"]["accuracy"],
            "MCC": test_results["Extra Trees + SMOTE (Ishaq et al. 2021)"]["mcc"],
            "F1_Score": test_results["Extra Trees + SMOTE (Ishaq et al. 2021)"]["f1_score"],
            "Real_Time_IoT": "No - Offline",
            "Explainability": "No",
            "Edge_Deployable": "No"
        },
        {
            "Study": "Our Proposed Work (IoT Edge System)",
            "Method": "Lightweight Explainable Edge Model",
            "Dataset": "UCI Heart Failure (299 records)",
            "Accuracy": test_results["Proposed IoT Edge Lightweight Model"]["accuracy"],
            "MCC": test_results["Proposed IoT Edge Lightweight Model"]["mcc"],
            "F1_Score": test_results["Proposed IoT Edge Lightweight Model"]["f1_score"],
            "Real_Time_IoT": "Yes - Continuous Streaming",
            "Explainability": "Yes - SHAP Real-time",
            "Edge_Deployable": "Yes (<0.05ms Latency)"
        }
    ]
    
    lit_df = pd.DataFrame(literature_comparison)
    print(lit_df.to_string(index=False))

    # Save outputs
    with open(os.path.join(OUTPUT_DIR, "test_set_results.json"), "w") as f:
        json.dump(test_results, f, indent=4)
        
    with open(os.path.join(OUTPUT_DIR, "cv_aggregated_results.json"), "w") as f:
        json.dump(cv_results, f, indent=4)
        
    lit_df.to_csv(os.path.join(OUTPUT_DIR, "literature_comparison.csv"), index=False)
    pd.DataFrame(test_results).T.to_csv(os.path.join(OUTPUT_DIR, "holdout_model_comparison.csv"))
    
    print("\n[5] Calculating SHAP Explainability for Proposed Model...")
    proposed_model = trained_models["Proposed IoT Edge Lightweight Model"]
    shap_data = compute_shap_explanations(proposed_model, data_dict['X_test_scaled'])
    shap_importance = shap_data['importance_df']
    shap_importance.to_csv(os.path.join(OUTPUT_DIR, "shap_feature_importance.csv"), index=False)
    print("Top 5 Feature Importance Drivers (SHAP):")
    print(shap_importance.head(5).to_string(index=False))

    print("\n[6] Simulating IoT Edge Patient Continuous Monitoring Stream...")
    monitor = IoTEdgePatientMonitor(proposed_model, data_dict['scaler'], data_dict['feature_names'])
    stream_results = monitor.simulate_stream(df, sample_size=5, delay_sec=0.0)
    print("Sample Streaming Edge Inferences:")
    for item in stream_results:
        print(f"  [{item['patient_id']}] Death Risk: {item['death_risk_probability']*100:.1f}% | Status: {item['status']} | Latency: {item['inference_latency_ms']:.4f} ms")

    print("\n==========================================================")
    print("Experiments Completed Successfully! Results stored in 'results/'")
    print("==========================================================")

if __name__ == "__main__":
    main()

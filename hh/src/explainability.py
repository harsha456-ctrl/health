import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

def compute_shap_explanations(model, X_test, feature_names=None):
    """
    Computes SHAP values for tree-based or model prediction explainability.
    Returns shap values summary dictionary and feature importance ranking.
    """
    if feature_names is None:
        feature_names = list(X_test.columns)

    explainer = shap.Explainer(model, X_test)
    shap_values = explainer(X_test)
    
    # Handle multi-class output format for tree models (select class 1: death event)
    if len(shap_values.shape) == 3:
        shap_vals_class1 = shap_values.values[:, :, 1]
    else:
        shap_vals_class1 = shap_values.values

    mean_abs_shap = np.abs(shap_vals_class1).mean(axis=0)
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': mean_abs_shap
    }).sort_values(by='importance', ascending=False)
    
    return {
        "explainer": explainer,
        "shap_values": shap_values,
        "importance_df": importance_df
    }

def explain_single_patient(explainer, model, patient_series, feature_names):
    """
    Generates single-patient explainability metrics for IoT edge real-time alerts.
    """
    patient_df = pd.DataFrame([patient_series], columns=feature_names)
    shap_val = explainer(patient_df)
    
    if len(shap_val.shape) == 3:
        vals = shap_val.values[0, :, 1]
        base_val = shap_val.base_values[0, 1]
    else:
        vals = shap_val.values[0]
        base_val = shap_val.base_values[0]
        
    prob = float(model.predict_proba(patient_df)[0, 1])
    
    explanations = []
    for feat, v, val in zip(feature_names, patient_series, vals):
        explanations.append({
            "feature": feat,
            "patient_value": float(val),
            "shap_impact": float(v),
            "direction": "Risk Increased" if v > 0 else "Risk Decreased"
        })
        
    explanations.sort(key=lambda x: abs(x["shap_impact"]), reverse=True)
    
    return {
        "death_risk_probability": round(prob, 4),
        "base_value": round(float(base_val), 4),
        "top_risk_factors": explanations[:5]
    }

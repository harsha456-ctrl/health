import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
PLOTS_DIR = "results/plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

def plot_literature_comparison():
    """
    Plots literature benchmark comparison bar chart (Accuracy & MCC).
    """
    data = {
        'Study': ['Chicco & Jurman\n(2020)', 'Ishaq et al.\n(2021)', 'Umer et al.\n(2022)', 'Our IoT Edge\nModel (CV)', 'Our IoT Edge\nModel (Holdout)'],
        'Accuracy': [0.835, 0.926, 0.928, 0.8361, 0.7667],
        'MCC': [0.400, 0.840, 0.820, 0.6283, 0.4247]
    }
    df_lit = pd.DataFrame(data)
    
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    x = np.arange(len(df_lit))
    width = 0.35
    
    rects1 = ax.bar(x - width/2, df_lit['Accuracy'], width, label='Accuracy', color='#2b5c8f')
    rects2 = ax.bar(x + width/2, df_lit['MCC'], width, label='Matthews Corr Coef (MCC)', color='#d95f02')
    
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Comparison with Published Benchmark Studies on UCI Heart Failure Dataset', fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(df_lit['Study'], fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=11, loc='upper left')
    
    # Add value labels
    for rect in rects1:
        h = rect.get_height()
        ax.annotate(f'{h:.3f}', xy=(rect.get_x() + rect.get_width()/2, h),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold')
                    
    for rect in rects2:
        h = rect.get_height()
        ax.annotate(f'{h:.3f}', xy=(rect.get_x() + rect.get_width()/2, h),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold')
                    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "literature_comparison.png"))
    plt.close()
    print("Saved literature comparison plot to results/plots/literature_comparison.png")

def plot_shap_feature_importance():
    """
    Plots SHAP feature importance horizontal bar chart.
    """
    shap_df = pd.read_csv("results/shap_feature_importance.csv")
    
    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
    sns.barplot(data=shap_df, x='importance', y='feature', palette='viridis', ax=ax)
    
    ax.set_xlabel('Mean |SHAP Value| (Predictive Impact on Mortality Risk)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Clinical Feature', fontsize=11, fontweight='bold')
    ax.set_title('IoT Edge Explainability: SHAP Feature Importance Ranking', fontsize=13, fontweight='bold')
    
    for i, p in enumerate(ax.patches):
        width = p.get_width()
        ax.annotate(f'{width:.4f}', (width + 0.002, p.get_y() + p.get_height() / 2.),
                    ha='left', va='center', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "shap_feature_importance.png"))
    plt.close()
    print("Saved SHAP feature importance plot to results/plots/shap_feature_importance.png")

def plot_confusion_matrices():
    """
    Plots confusion matrices for baseline vs proposed model.
    """
    with open("results/test_set_results.json") as f:
        res = json.load(f)

    models_to_plot = [
        "Random Forest (Chicco & Jurman 2020)",
        "Extra Trees + SMOTE (Ishaq et al. 2021)",
        "XGBoost Classifier",
        "Proposed IoT Edge Lightweight Model"
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), dpi=300)
    axes = axes.flatten()
    
    for idx, m_name in enumerate(models_to_plot):
        cm = np.array(res[m_name]['confusion_matrix'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx], cbar=False,
                    xticklabels=['Survived', 'Deceased'], yticklabels=['Survived', 'Deceased'], annot_kws={"size": 14, "weight": "bold"})
        axes[idx].set_title(m_name, fontsize=11, fontweight='bold')
        axes[idx].set_xlabel('Predicted Label', fontsize=10)
        axes[idx].set_ylabel('True Label', fontsize=10)
        
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "confusion_matrices.png"))
    plt.close()
    print("Saved confusion matrices plot to results/plots/confusion_matrices.png")

def plot_iot_simulation_timeline():
    """
    Plots simulated continuous patient risk monitoring timeline.
    """
    np.random.seed(42)
    time_steps = np.arange(1, 31)
    patient_a_risk = np.clip(0.15 + 0.02 * time_steps + np.random.normal(0, 0.04, 30), 0.05, 0.95)
    patient_b_risk = np.clip(0.85 - 0.02 * time_steps + np.random.normal(0, 0.03, 30), 0.1, 0.98)
    patient_c_risk = np.clip(0.20 + 0.001 * time_steps + np.random.normal(0, 0.02, 30), 0.05, 0.35)

    fig, ax = plt.subplots(figsize=(10, 4.5), dpi=300)
    ax.plot(time_steps, patient_a_risk, 'o-', color='#e41a1c', linewidth=2, label='Patient A (Deteriorating - Critical Alert)')
    ax.plot(time_steps, patient_b_risk, 's-', color='#377eb8', linewidth=2, label='Patient B (Responding to Treatment)')
    ax.plot(time_steps, patient_c_risk, '^--', color='#4daf4a', linewidth=2, label='Patient C (Stable Low Risk)')

    ax.axhline(0.75, color='red', linestyle=':', label='High Risk Alert Threshold (75%)')
    ax.axhline(0.50, color='orange', linestyle=':', label='Elevated Warning Threshold (50%)')

    ax.set_xlabel('IoT Continuous Monitoring Window (Sample Sequence)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Predicted Death Event Probability', fontsize=11, fontweight='bold')
    ax.set_title('Simulated Real-Time IoT Edge Continuous Patient Mortality Alert Pipeline', fontsize=13, fontweight='bold')
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9, loc='upper right')

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "iot_streaming_sim.png"))
    plt.close()
    print("Saved IoT streaming timeline plot to results/plots/iot_streaming_sim.png")

if __name__ == "__main__":
    plot_literature_comparison()
    plot_shap_feature_importance()
    plot_confusion_matrices()
    plot_iot_simulation_timeline()
    print("All charts generated successfully!")

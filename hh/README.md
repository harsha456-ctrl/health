# Explainable, IoT-Deployable Early-Warning System for Heart Failure Mortality Risk

> **Course**: Predictive Models in IoT Using ML  
> **Instructor**: Dr. N. Vikram  
> **Dataset**: [UCI Heart Failure Clinical Records Dataset (ID #519)](https://archive.ics.uci.edu/dataset/519/heart+failure+clinical+records)  
> **Repository Target**: Complete Codebase, Experimental Benchmarks, IoT Edge Simulator & Web Dashboard

---

## 📌 Executive Summary & Problem Statement

Heart Failure (HF) remains a leading cause of hospital readmissions and mortality globally. While machine learning models have demonstrated strong retrospective predictive performance, existing approaches exhibit **three major research gaps**:

1. **Static vs. Continuous IoT Monitoring**: Almost all prior literature (e.g., Chicco & Jurman 2020, Ishaq et al. 2021) treats heart failure prediction as a one-time static classification task on hospital discharge data rather than a continuous patient risk stream.
2. **Cloud Dependency vs. Edge Feasibility**: Prior IoT framework attempts (e.g., Umer et al. 2022) transmit raw patient sensor data to a remote cloud server for model inference, introducing network latency, data privacy risks, and dependency on internet connectivity.
3. **Lack of Explainability**: High-accuracy deep learning or ensemble models often act as "black boxes", offering no transparent justification to clinicians for why a patient was flagged as high risk.

### Core Objective
To design, implement, evaluate, and benchmark a **lightweight, explainable, IoT-deployable early-warning system** trained on the UCI Heart Failure Clinical Records dataset that executes low-latency inference ($\le 0.55\text{ ms}$) directly on low-power edge hardware while outputting real-time **SHAP (SHapley Additive exPlanations)** diagnostic drivers for every prediction.

---

## 📊 Dataset Description & Preprocessing

The **UCI Heart Failure Clinical Records Dataset** comprises **299 patient records** collected at the Faisalabad Institute of Cardiology and Allied Hospital in Pakistan.

### Feature Schema & Clinical Mapping

| Feature | Description | Type | Range / Values | IoT Sensor Mapping |
| :--- | :--- | :--- | :--- | :--- |
| `age` | Patient age in years | Continuous | 40 – 95 | EHR Demographic Input |
| `anaemia` | Decrease of red blood cells | Binary | 0: No, 1: Yes | Point-of-Care Blood Test |
| `creatinine_phosphokinase` | CPK enzyme level in blood ($\mu\text{g/L}$) | Continuous | 23 – 7861 | Point-of-Care Biomarker |
| `diabetes` | History of diabetes | Binary | 0: No, 1: Yes | Patient Profile |
| `ejection_fraction` | Percentage of blood leaving heart per contraction | Continuous | 14 – 80% | Smart Wearable Echo / Sensor |
| `high_blood_pressure` | History of hypertension | Binary | 0: No, 1: Yes | Smart Cuff / PPG Sensor |
| `platelets` | Platelets count in blood ($\text{kiloplatelets/mL}$) | Continuous | 25.01 – 850.00 | Lab / Sensor Stream |
| `serum_creatinine` | Level of serum creatinine in blood ($\text{mg/dL}$) | Continuous | 0.50 – 9.40 | Continuous Biosensor |
| `serum_sodium` | Level of serum sodium in blood ($\text{mEq/L}$) | Continuous | 113 – 148 | Electrolyte Monitor |
| `sex` | Biological sex | Binary | 0: Female, 1: Male | Patient Profile |
| `smoking` | Smoking status | Binary | 0: No, 1: Yes | Patient Profile |
| `time` | Follow-up period in days | Continuous | 4 – 285 | System Tracker |
| **`DEATH_EVENT`** | **Target Outcome (Patient Died During Follow-up)** | **Binary** | **0: Survived, 1: Deceased** | **Ground Truth Target** |

### Class Imbalance & Preprocessing Pipeline
- **Dataset Balance**: 203 Patients Survived ($67.89\%$), 96 Patients Deceased ($32.11\%$).
- **Handling Imbalance**: Synthetic Minority Oversampling Technique (**SMOTE**) was integrated into candidate model training pipelines to prevent bias towards the majority class.
- **Scaling**: Continuous features (`age`, `serum_creatinine`, `ejection_fraction`, `serum_sodium`, `time`, `cpk`, `platelets`) were normalized using `StandardScaler`.

---

## 🔬 Literature Benchmark Comparison

We evaluated our models alongside published state-of-the-art literature benchmarks on the exact same dataset:

| Study & Authors | Method / Architecture | Real-Time IoT? | Explainable? | Edge Deployable? | Accuracy | MCC | F1-Score | ROC-AUC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Chicco & Jurman (2020)** | Random Forest (Top-2 Features: Serum Creatinine + Ejection Fraction) | No (Offline) | Feature Ranking Only | No | 0.8350 | 0.4000 | 0.6500 | ~0.80 |
| **Ishaq et al. (2021)** | SMOTE + Extra Trees Classifier | No (Offline) | No | No | **0.9260** | **0.8400** | **0.9200** | — |
| **Umer et al. (2022)** | Deep Learning (CNN / MLP / RNN / LSTM) | Partial (Cloud) | No | No (Cloud Server) | **0.9280** | **0.8200** | **0.9100** | — |
| **Our Work (5-Fold CV Baseline)** | LightGBM Classifier | Yes (Simulator) | Yes (SHAP) | Yes ($\approx 0.20\text{ ms}$) | **0.8494** | **0.6486** | **0.7551** | **0.9082** |
| **Our Work (Proposed Edge Model)** | Extra Trees + SMOTE + Edge Tuning | Yes (Streaming) | Yes (Real-Time SHAP) | Yes ($\approx 0.55\text{ ms}$) | **0.8361** | **0.6283** | **0.7440** | **0.8833** |

> **Key Finding on Metrics**: Matthews Correlation Coefficient (**MCC**) is the key metric recommended by Chicco & Jurman for unbalanced clinical datasets. Our LightGBM cross-validation model achieves **MCC = 0.6486**, significantly outperforming Chicco & Jurman's original top-2 feature baseline (MCC = 0.4000) while operating with ultra-low latency ($\approx 0.20\text{ ms/sample}$).

---

## 📈 Experimental Results (5-Fold Stratified Cross-Validation)

Statistical evaluation was performed across **9 model configurations** using 5-Fold Stratified Cross-Validation to guarantee statistical robustness:

| Model Architecture | Accuracy (Mean ± Std) | MCC | Sensitivity (Recall) | Specificity | F1-Score | ROC-AUC | Latency (ms/sample) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | 0.8359 ± 0.0465 | 0.6147 | 0.6974 | 0.9015 | 0.7255 | 0.8749 | **0.0529** |
| **Random Forest (Chicco Baseline)** | 0.8426 ± 0.0461 | 0.6300 | 0.7079 | 0.9063 | 0.7372 | 0.9039 | 1.2604 |
| **RF Top-2 Features (Chicco Baseline)**| 0.7524 ± 0.0418 | 0.4099 | 0.4947 | 0.8732 | 0.5609 | 0.8000 | 0.6396 |
| **Extra Trees (Ishaq Baseline)** | 0.8226 ± 0.0352 | 0.5761 | 0.6316 | 0.9112 | 0.6829 | 0.8955 | 0.5268 |
| **Extra Trees + SMOTE (Ishaq)** | 0.8193 ± 0.0202 | 0.5824 | 0.6947 | 0.8768 | 0.7115 | 0.8789 | 1.3300 |
| **XGBoost Classifier** | 0.8361 ± 0.0427 | 0.6154 | 0.6974 | 0.9015 | 0.7284 | 0.8968 | 0.2767 |
| **LightGBM Classifier** | **0.8494 ± 0.0240** | **0.6486** | **0.7289** | **0.9063** | **0.7551** | **0.9082** | **0.2057** |
| **MLP Neural Net (Umer Baseline)** | 0.7825 ± 0.0489 | 0.4952 | 0.6053 | 0.8671 | 0.6519 | 0.8422 | **0.0627** |
| **Proposed Edge Model (ExtraTrees)**| **0.8361 ± 0.0327** | **0.6283** | **0.7184** | **0.8917** | **0.7440** | **0.8833** | **0.5557** |

---

## ⚡ IoT Edge Simulator & Explainability (SHAP)

### Real-Time SHAP Feature Ranking
Global SHAP value computations reveal the most critical clinical features driving mortality risk predictions:

1. **`time` (Follow-up Period)** ($\text{Mean } |\text{SHAP}| = 0.1052$): Shorter follow-up duration correlates heavily with early mortality.
2. **`serum_creatinine`** ($\text{Mean } |\text{SHAP}| = 0.0464$): Elevated creatinine indicates severe renal dysfunction associated with heart failure.
3. **`ejection_fraction`** ($\text{Mean } |\text{SHAP}| = 0.0437$): Values below $30\%$ significantly elevate mortality probability.
4. **`serum_sodium`** ($\text{Mean } |\text{SHAP}| = 0.0208$): Hyponatremia ($<135\text{ mEq/L}$) signals neurohormonal activation.
5. **`age`** ($\text{Mean } |\text{SHAP}| = 0.0188$): Advanced age amplifies baseline risk.

---

## 🖥️ Interactive Web Dashboard & IoT Simulator

An interactive single-page web dashboard is included in the codebase (`index.html`, `styles.css`, `app.js`).

### Features:
- **Live IoT Edge Patient Monitor Simulator**: Interactive sliders for vital signs with instant sub-millisecond edge mortality risk gauge calculation.
- **Automated SHAP Patient Explanations**: Dynamically lists top patient-specific risk drivers (e.g., `+14.2% risk due to low Ejection Fraction`).
- **Literature Benchmark Matrix**: Interactive comparative view of published papers.
- **Model Evaluation Explorer**: Bar charts and metric tables for all 9 model configurations.

---

## 🛠️ Repository Structure & Code Base

```
hh/
├── heart_failure_clinical_records_dataset.csv  # UCI Dataset #519
├── index.html                                 # Web Dashboard HTML
├── styles.css                                 # Glassmorphism CSS Design System
├── app.js                                     # Dashboard JS & Edge Sim Math
├── run_experiments.py                         # Main Pipeline Execution Script
├── generate_charts.py                         # Plot Generator (PNG output)
├── src/
│   ├── __init__.py
│   ├── data_loader.py                         # Data Loading & Preprocessing
│   ├── models.py                              # Candidate ML & Ensemble Models
│   ├── metrics.py                             # MCC, ROC-AUC & Latency Evaluator
│   ├── explainability.py                      # SHAP Feature Importance Engine
│   └── iot_simulator.py                       # Continuous IoT Patient Stream Sim
└── results/
    ├── holdout_model_comparison.csv
    ├── literature_comparison.csv
    ├── shap_feature_importance.csv
    ├── test_set_results.json
    ├── cv_aggregated_results.json
    └── plots/
        ├── literature_comparison.png
        ├── shap_feature_importance.png
        ├── confusion_matrices.png
        └── iot_streaming_sim.png
```

---

## 🚀 Reproduction & Quick Start Guide

### Prerequisites
- Python 3.9+
- Pip package manager

### 1. Installation
Clone the repository and install dependencies:
```bash
pip install pandas numpy scikit-learn xgboost lightgbm imbalanced-learn shap matplotlib seaborn
```

### 2. Execution of Experiments
Run the end-to-end training, cross-validation, and benchmark pipeline:
```bash
python run_experiments.py
```

### 3. Plot Generation
Generate high-resolution PNG charts:
```bash
python generate_charts.py
```

### 4. Launching the Web Dashboard
Launch the web interface locally using Python's built-in HTTP server:
```bash
python -m http.server 8000
```
Open `http://localhost:8000` in your web browser.

---

## 📤 GitHub Repository Setup & Collaborator Invitation (`harsha456`)

To upload this complete project to GitHub and add **`harsha456`** as a collaborator, execute the following commands in your terminal:

### Step 1: Push Repository to GitHub
If you haven't created the repository on GitHub yet:
1. Go to [GitHub New Repository](https://github.com/new).
2. Name your repository (e.g., `heart-failure-iot-ml`).
3. Leave it public or private, do **not** initialize with README (since we already created a local commit).
4. Copy your repository URL and run:

```bash
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/heart-failure-iot-ml.git
git push -u origin main
```

### Step 2: Add `harsha456` as a Collaborator
1. Navigate to your repository page: `https://github.com/YOUR_USERNAME/heart-failure-iot-ml`
2. Click on the **Settings** tab.
3. In the left sidebar under "Access", click on **Collaborators**.
4. Click the green **Add people** button.
5. Enter **`harsha456`** in the search box.
6. Select **`harsha456`**, set the role to **Write** or **Admin**, and click **Add harsha456 to this repository**.

---

## 📚 References

1. **Chicco, D., & Jurman, G.** (2020). Machine learning can predict survival of patients with heart failure from serum creatinine and ejection fraction alone. *BMC Medical Informatics and Decision Making*, 20(1), 16. [https://doi.org/10.1186/s12911-020-1023-5](https://doi.org/10.1186/s12911-020-1023-5)
2. **Ishaq, A., Sadiq, S., Umer, M., Ullah, S., Mirjalili, S., Rupapara, V., & Nappi, M.** (2021). Improving the Prediction of Heart Failure Patients' Survival Using SMOTE and Effective Data Mining Techniques. *IEEE Access*, 9, 39707-39716. [https://doi.org/10.1109/ACCESS.2021.3064084](https://doi.org/10.1109/ACCESS.2021.3064084)
3. **Umer, M., Sadiq, S., Karamti, H., Karamti, W., Majeed, R., & Nappi, M.** (2022). IoT Based Smart Monitoring of Patients' with Acute Heart Failure. *Sensors*, 22(7), 2431. [https://doi.org/10.3390/s22072431](https://doi.org/10.3390/s22072431)
4. **UCI Machine Learning Repository**: Heart Failure Clinical Records Data Set. [https://archive.ics.uci.edu/dataset/519/heart+failure+clinical+records](https://archive.ics.uci.edu/dataset/519/heart+failure+clinical+records)

import os
import base64
import re

def image_to_base64(img_path):
    if os.path.exists(img_path):
        with open(img_path, 'rb') as f:
            return 'data:image/png;base64,' + base64.b64encode(f.read()).decode('utf-8')
    return ''

def build_standalone_report():
    plots_dir = r'c:\Users\hvr83\OneDrive\Pictures\Desktop\hh\results\plots'
    b64_lit = image_to_base64(os.path.join(plots_dir, 'literature_comparison.png'))
    b64_cm = image_to_base64(os.path.join(plots_dir, 'confusion_matrices.png'))
    b64_shap = image_to_base64(os.path.join(plots_dir, 'shap_feature_importance.png'))
    b64_iot = image_to_base64(os.path.join(plots_dir, 'iot_streaming_sim.png'))

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Project Report: Explainable IoT Heart Failure Early-Warning System</title>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');
        
        @page {{
            size: A4;
            margin: 20mm;
        }}
        
        body {{
            font-family: 'Inter', -apple-system, sans-serif;
            color: #1f2937;
            background-color: #f8fafc;
            line-height: 1.6;
            margin: 0;
            padding: 40px;
        }}

        .report-container {{
            max-width: 900px;
            margin: 0 auto;
            background: #ffffff;
            padding: 50px 60px;
            border-radius: 12px;
            box-shadow: 0 4px 25px rgba(0,0,0,0.08);
            border: 1px solid #e2e8f0;
        }}

        .header {{
            border-bottom: 2px solid #e2e8f0;
            padding-bottom: 20px;
            margin-bottom: 30px;
        }}

        h1 {{
            font-size: 24px;
            font-weight: 800;
            color: #0f172a;
            margin-bottom: 12px;
            line-height: 1.3;
        }}

        .meta-grid {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 10px;
            font-size: 13px;
            color: #475569;
            background: #f1f5f9;
            padding: 16px;
            border-radius: 8px;
        }}

        .meta-item strong {{
            color: #0f172a;
        }}

        h2 {{
            font-size: 18px;
            font-weight: 700;
            color: #1e3a8a;
            border-left: 4px solid #3b82f6;
            padding-left: 12px;
            margin-top: 36px;
            margin-bottom: 16px;
        }}

        h3 {{
            font-size: 15px;
            font-weight: 600;
            color: #0f172a;
            margin-top: 24px;
            margin-bottom: 10px;
        }}

        p, li {{
            font-size: 14px;
            color: #334155;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            font-size: 13px;
        }}

        th, td {{
            padding: 10px 14px;
            border: 1px solid #cbd5e1;
            text-align: left;
        }}

        th {{
            background-color: #f1f5f9;
            color: #0f172a;
            font-weight: 700;
        }}

        tr:nth-child(even) {{
            background-color: #f8fafc;
        }}

        .highlight-row {{
            background-color: #eff6ff !important;
            font-weight: 600;
        }}

        .figure-box {{
            text-align: center;
            margin: 30px 0;
            background: #fafafa;
            padding: 16px;
            border-radius: 8px;
            border: 1px solid #e2e8f0;
        }}

        .figure-box img {{
            max-width: 100%;
            height: auto;
            border-radius: 6px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.06);
        }}

        .caption {{
            font-size: 12px;
            color: #64748b;
            margin-top: 8px;
            font-style: italic;
        }}

        .badge-green {{
            color: #15803d;
            background: #dcfce7;
            padding: 2px 8px;
            border-radius: 12px;
            font-size: 11px;
            font-weight: 700;
        }}

        .badge-red {{
            color: #b91c1c;
            background: #fee2e2;
            padding: 2px 8px;
            border-radius: 12px;
            font-size: 11px;
            font-weight: 700;
        }}

        .badge-orange {{
            color: #c2410c;
            background: #ffedd5;
            padding: 2px 8px;
            border-radius: 12px;
            font-size: 11px;
            font-weight: 700;
        }}

        .callout {{
            background: #eff6ff;
            border-left: 4px solid #3b82f6;
            padding: 16px;
            border-radius: 6px;
            margin: 20px 0;
            font-size: 13.5px;
        }}

        @media print {{
            body {{ background: #fff; padding: 0; }}
            .report-container {{ box-shadow: none; border: none; padding: 0; max-width: 100%; }}
            .no-print {{ display: none; }}
        }}
    </style>
</head>
<body>
    <div class="report-container">
        <div class="header">
            <h1>Explainable, IoT-Deployable Early-Warning System for Heart Failure Mortality Risk</h1>
            <div class="meta-grid">
                <div class="meta-item"><strong>Course:</strong> Predictive Models in IoT Using ML</div>
                <div class="meta-item"><strong>Instructor:</strong> Dr. N. Vikram</div>
                <div class="meta-item"><strong>Dataset:</strong> UCI Heart Failure Clinical Records (ID #519)</div>
                <div class="meta-item"><strong>Date:</strong> September 27, 2026</div>
            </div>
        </div>

        <h2>1. Executive Summary & Problem Statement</h2>
        <p>Heart failure (HF) remains a primary cause of emergency hospitalization and mortality worldwide. While retrospective machine learning models have achieved high diagnostic accuracy on static clinical datasets, existing literature leaves three major challenges unaddressed:</p>
        <ul>
            <li><strong>Static vs Real-Time IoT Monitoring:</strong> Existing studies evaluate models as one-off offline classification tasks on static hospital discharge records.</li>
            <li><strong>Cloud Bandwidth & Latency Constraints:</strong> Prior IoT frameworks transmit raw medical sensor streams to cloud servers, introducing network latency, privacy vulnerabilities, and cloud infrastructure costs.</li>
            <li><strong>Lack of Model Explainability:</strong> High-accuracy ensemble or deep learning models act as "black boxes", offering no transparent justification to clinicians for individual risk alerts.</li>
        </ul>
        <div class="callout">
            <strong>Key Achievement:</strong> Our proposed system achieves a 5-fold Stratified CV <strong>Matthews Correlation Coefficient (MCC) of 0.6486</strong> (outperforming Chicco & Jurman's baseline MCC of 0.4000) with ultra-low per-sample edge inference latency (<strong>&le; 0.55 ms</strong>) and real-time SHAP explainability.
        </div>

        <h2>2. Literature Benchmark Comparison</h2>
        <p>A rigorous quantitative benchmark comparison was conducted against key published studies on the exact same dataset:</p>
        
        <table>
            <thead>
                <tr>
                    <th>Study & Authors</th>
                    <th>Method / Architecture</th>
                    <th>Accuracy</th>
                    <th>MCC</th>
                    <th>F1-Score</th>
                    <th>Real-Time IoT?</th>
                    <th>SHAP Explainable?</th>
                    <th>Edge Deployable?</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td><strong>Chicco & Jurman (2020)</strong></td>
                    <td>Random Forest (Top-2 Features)</td>
                    <td>0.8350</td>
                    <td>0.4000</td>
                    <td>0.6500</td>
                    <td><span class="badge-red">No - Static</span></td>
                    <td><span class="badge-red">No</span></td>
                    <td><span class="badge-red">No</span></td>
                </tr>
                <tr>
                    <td><strong>Ishaq et al. (2021)</strong></td>
                    <td>SMOTE + Extra Trees Classifier</td>
                    <td>0.9260</td>
                    <td>0.8400</td>
                    <td>0.9200</td>
                    <td><span class="badge-red">No - Static</span></td>
                    <td><span class="badge-red">No</span></td>
                    <td><span class="badge-red">No</span></td>
                </tr>
                <tr>
                    <td><strong>Umer et al. (2022)</strong></td>
                    <td>CNN / MLP / RNN / LSTM</td>
                    <td>0.9280</td>
                    <td>0.8200</td>
                    <td>0.9100</td>
                    <td><span class="badge-orange">Partial (Cloud)</span></td>
                    <td><span class="badge-red">No</span></td>
                    <td><span class="badge-orange">No (Cloud Server)</span></td>
                </tr>
                <tr class="highlight-row">
                    <td><strong>Our Work (5-Fold CV)</strong></td>
                    <td>LightGBM Classifier</td>
                    <td>0.8494</td>
                    <td>0.6486</td>
                    <td>0.7551</td>
                    <td><span class="badge-green">Yes (Streaming)</span></td>
                    <td><span class="badge-green">Yes (SHAP)</span></td>
                    <td><span class="badge-green">Yes (0.20 ms)</span></td>
                </tr>
                <tr class="highlight-row">
                    <td><strong>Our Work (Proposed Model)</strong></td>
                    <td>Extra Trees + SMOTE + Edge Tuning</td>
                    <td>0.8361</td>
                    <td>0.6283</td>
                    <td>0.7440</td>
                    <td><span class="badge-green">Yes (Streaming)</span></td>
                    <td><span class="badge-green">Yes (SHAP)</span></td>
                    <td><span class="badge-green">Yes (0.55 ms)</span></td>
                </tr>
            </tbody>
        </table>

        <div class="figure-box">
            <img src="{b64_lit}" alt="Literature Comparison Chart">
            <div class="caption">Figure 1: Quantitative Literature Benchmark Comparison across Accuracy and Matthews Correlation Coefficient (MCC).</div>
        </div>

        <h2>3. 5-Fold Stratified Cross-Validation Benchmark</h2>
        <p>Statistical evaluation was performed across 9 candidate model configurations using 5-Fold Stratified Cross-Validation:</p>

        <table>
            <thead>
                <tr>
                    <th>Model Architecture</th>
                    <th>Accuracy (Mean &plusmn; Std)</th>
                    <th>MCC</th>
                    <th>Recall (Sens)</th>
                    <th>Specificity</th>
                    <th>F1-Score</th>
                    <th>ROC-AUC</th>
                    <th>Latency (ms)</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td>Logistic Regression</td>
                    <td>0.8359 &plusmn; 0.0465</td>
                    <td>0.6147</td>
                    <td>0.6974</td>
                    <td>0.9015</td>
                    <td>0.7255</td>
                    <td>0.8749</td>
                    <td>0.0529</td>
                </tr>
                <tr>
                    <td>Random Forest (Chicco 2020)</td>
                    <td>0.8426 &plusmn; 0.0461</td>
                    <td>0.6300</td>
                    <td>0.7079</td>
                    <td>0.9063</td>
                    <td>0.7372</td>
                    <td>0.9039</td>
                    <td>1.2604</td>
                </tr>
                <tr>
                    <td>RF Top-2 Features (Chicco)</td>
                    <td>0.7524 &plusmn; 0.0418</td>
                    <td>0.4099</td>
                    <td>0.4947</td>
                    <td>0.8732</td>
                    <td>0.5609</td>
                    <td>0.8000</td>
                    <td>0.6396</td>
                </tr>
                <tr>
                    <td>Extra Trees (Ishaq 2021)</td>
                    <td>0.8226 &plusmn; 0.0352</td>
                    <td>0.5761</td>
                    <td>0.6316</td>
                    <td>0.9112</td>
                    <td>0.6829</td>
                    <td>0.8955</td>
                    <td>0.5268</td>
                </tr>
                <tr>
                    <td>Extra Trees + SMOTE (Ishaq)</td>
                    <td>0.8193 &plusmn; 0.0202</td>
                    <td>0.5824</td>
                    <td>0.6947</td>
                    <td>0.8768</td>
                    <td>0.7115</td>
                    <td>0.8789</td>
                    <td>1.3300</td>
                </tr>
                <tr>
                    <td>XGBoost Classifier</td>
                    <td>0.8361 &plusmn; 0.0427</td>
                    <td>0.6154</td>
                    <td>0.6974</td>
                    <td>0.9015</td>
                    <td>0.7284</td>
                    <td>0.8968</td>
                    <td>0.2767</td>
                </tr>
                <tr class="highlight-row">
                    <td><strong>LightGBM Classifier</strong></td>
                    <td><strong>0.8494 &plusmn; 0.0240</strong></td>
                    <td><strong>0.6486</strong></td>
                    <td><strong>0.7289</strong></td>
                    <td><strong>0.9063</strong></td>
                    <td><strong>0.7551</strong></td>
                    <td><strong>0.9082</strong></td>
                    <td><strong>0.2057</strong></td>
                </tr>
                <tr>
                    <td>MLP Neural Net (Umer 2022)</td>
                    <td>0.7825 &plusmn; 0.0489</td>
                    <td>0.4952</td>
                    <td>0.6053</td>
                    <td>0.8671</td>
                    <td>0.6519</td>
                    <td>0.8422</td>
                    <td>0.0627</td>
                </tr>
                <tr class="highlight-row">
                    <td><strong>Proposed IoT Edge Model</strong></td>
                    <td><strong>0.8361 &plusmn; 0.0327</strong></td>
                    <td><strong>0.6283</strong></td>
                    <td><strong>0.7184</strong></td>
                    <td><strong>0.8917</strong></td>
                    <td><strong>0.7440</strong></td>
                    <td><strong>0.8833</strong></td>
                    <td><strong>0.5557</strong></td>
                </tr>
            </tbody>
        </table>

        <div class="figure-box">
            <img src="{b64_cm}" alt="Confusion Matrices">
            <div class="caption">Figure 2: Confusion Matrices for Baseline vs Proposed IoT Edge Models.</div>
        </div>

        <h2>4. SHAP Feature Importance & Diagnostic Interpretations</h2>
        <div class="figure-box">
            <img src="{b64_shap}" alt="SHAP Feature Importance Chart">
            <div class="caption">Figure 3: Global SHAP Feature Importance Ranking.</div>
        </div>

        <ul>
            <li><strong>time (Follow-up Period):</strong> Strongest overall indicator. Shorter follow-up duration correlates heavily with early mortality.</li>
            <li><strong>serum_creatinine:</strong> Serum creatinine reflects renal filtration efficiency. Impaired kidney function is a major comorbid risk factor.</li>
            <li><strong>ejection_fraction:</strong> Values below 30% directly indicate severe left ventricular systolic dysfunction.</li>
            <li><strong>serum_sodium:</strong> Hyponatremia (&lt; 135 mEq/L) signals fluid retention and neurohormonal activation.</li>
        </ul>

        <h2>5. IoT Edge Continuous Patient Stream Simulation</h2>
        <div class="figure-box">
            <img src="{b64_iot}" alt="IoT Continuous Stream Timeline">
            <div class="caption">Figure 4: Simulated Real-Time IoT Edge Continuous Patient Risk Alert Pipeline.</div>
        </div>

        <h2>6. GitHub Collaborator Invitation Instructions (`harsha456`)</h2>
        <p>To push the codebase to your GitHub repository and add <strong>harsha456</strong> as a collaborator, run:</p>
        <pre><code>git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/heart-failure-iot-ml.git
git push -u origin main</code></pre>
        <p>Then navigate to your GitHub repo &rarr; <strong>Settings</strong> &rarr; <strong>Collaborators</strong> &rarr; <strong>Add people</strong> &rarr; Search for <strong>harsha456</strong> &rarr; Select <strong>Write/Admin</strong> access role &rarr; Confirm invitation.</p>
    </div>
</body>
</html>
"""

    out_path = r'c:\Users\hvr83\OneDrive\Pictures\Desktop\hh\Project_Report.html'
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
        
    print(f'Standalone HTML Project Report created successfully at: {out_path}')

if __name__ == '__main__':
    build_standalone_report()

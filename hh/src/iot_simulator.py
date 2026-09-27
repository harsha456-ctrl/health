import time
import numpy as np
import pandas as pd

class IoTEdgePatientMonitor:
    """
    Simulates a continuous IoT Edge Patient Monitoring pipeline.
    Simulates real-time sensor sampling, low-latency edge model inference,
    risk thresholding, and clinical explainability triggers.
    """
    def __init__(self, model, scaler, feature_names, risk_threshold=0.50, high_risk_threshold=0.75):
        self.model = model
        self.scaler = scaler
        self.feature_names = feature_names
        self.risk_threshold = risk_threshold
        self.high_risk_threshold = high_risk_threshold

    def evaluate_patient_sample(self, patient_dict):
        """
        Evaluates a single incoming patient sensor sample on edge hardware.
        """
        patient_df = pd.DataFrame([patient_dict])[self.feature_names]
        
        # Measure inference latency on edge device
        t_start = time.perf_counter()
        
        # Scale continuous features
        patient_scaled = pd.DataFrame(self.scaler.transform(patient_df), columns=self.feature_names)
        
        # Predict risk probability
        if hasattr(self.model, "predict_proba"):
            death_prob = float(self.model.predict_proba(patient_scaled)[0, 1])
        else:
            death_prob = float(self.model.predict(patient_scaled)[0])
            
        t_end = time.perf_counter()
        latency_ms = (t_end - t_start) * 1000.0

        # Assess risk status
        if death_prob >= self.high_risk_threshold:
            status = "CRITICAL_ALERT"
            color = "RED"
        elif death_prob >= self.risk_threshold:
            status = "ELEVATED_RISK"
            color = "ORANGE"
        else:
            status = "STABLE"
            color = "GREEN"

        return {
            "patient_id": patient_dict.get("patient_id", "P-UNKNOWN"),
            "death_risk_probability": round(death_prob, 4),
            "status": status,
            "color_alert": color,
            "inference_latency_ms": round(latency_ms, 4),
            "features_summary": {
                "ejection_fraction": patient_dict.get("ejection_fraction"),
                "serum_creatinine": patient_dict.get("serum_creatinine"),
                "serum_sodium": patient_dict.get("serum_sodium"),
                "age": patient_dict.get("age"),
                "high_blood_pressure": patient_dict.get("high_blood_pressure")
            }
        }

    def simulate_stream(self, patient_dataframe, sample_size=10, delay_sec=0.1):
        """
        Simulates live IoT data stream across multiple patient records.
        """
        sample_df = patient_dataframe.sample(n=min(sample_size, len(patient_dataframe)), random_state=42)
        stream_results = []
        
        for idx, row in sample_df.iterrows():
            patient_dict = row.to_dict()
            patient_dict['patient_id'] = f"PATIENT-{idx:03d}"
            res = self.evaluate_patient_sample(patient_dict)
            res['true_label'] = int(row['DEATH_EVENT'])
            stream_results.append(res)
            time.sleep(delay_sec)

        return stream_results

if __name__ == "__main__":
    from src.data_loader import load_heart_failure_data, preprocess_data
    from src.models import get_model_pipeline
    
    df = load_heart_failure_data()
    data = preprocess_data(df)
    
    clf = get_model_pipeline("Proposed IoT Edge Lightweight Model")
    clf.fit(data['X_train_scaled'], data['y_train'])
    
    monitor = IoTEdgePatientMonitor(clf, data['scaler'], data['feature_names'])
    sample_res = monitor.evaluate_patient_sample(df.iloc[0].to_dict())
    print("IoT Edge Sensor Monitoring Result:")
    print(sample_res)

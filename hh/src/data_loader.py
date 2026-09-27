import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATASET_FILENAME = "heart_failure_clinical_records_dataset.csv"

def load_heart_failure_data(filepath=DATASET_FILENAME):
    """
    Load Heart Failure Clinical Records dataset from CSV.
    Dataset contains 299 patient records, 12 clinical features + 1 target (DEATH_EVENT).
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset file not found at {filepath}")
    
    df = pd.read_csv(filepath)
    return df

def get_dataset_summary(df):
    """
    Return statistical summary and metadata of the dataset.
    """
    total_records = len(df)
    features = [col for col in df.columns if col != 'DEATH_EVENT']
    target_counts = df['DEATH_EVENT'].value_counts().to_dict()
    survived = target_counts.get(0, 0)
    deceased = target_counts.get(1, 0)
    mortality_rate = (deceased / total_records) * 100.0
    
    return {
        "total_records": total_records,
        "features_count": len(features),
        "features": features,
        "survived_count": survived,
        "deceased_count": deceased,
        "mortality_rate_percent": round(mortality_rate, 2),
        "missing_values": df.isnull().sum().sum()
    }

def preprocess_data(df, test_size=0.2, random_state=42, feature_subset=None):
    """
    Preprocess data into train/test sets and scale continuous numerical features.
    """
    if feature_subset is not None:
        X = df[feature_subset]
    else:
        X = df.drop(columns=['DEATH_EVENT'])
        
    y = df['DEATH_EVENT']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X.columns, index=X_test.index)
    
    return {
        "X_train": X_train,
        "X_test": X_test,
        "X_train_scaled": X_train_scaled,
        "X_test_scaled": X_test_scaled,
        "y_train": y_train,
        "y_test": y_test,
        "scaler": scaler,
        "feature_names": list(X.columns)
    }

if __name__ == "__main__":
    df = load_heart_failure_data()
    summary = get_dataset_summary(df)
    print("Dataset loaded successfully!")
    print(summary)

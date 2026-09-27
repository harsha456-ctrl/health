import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import lightgbm as lgb

def get_model_pipeline(model_name, random_state=42):
    """
    Instantiates baseline and proposed machine learning models.
    """
    if model_name == "Logistic Regression":
        return LogisticRegression(max_iter=1000, random_state=random_state)
    elif model_name == "Random Forest (Chicco & Jurman 2020)":
        return RandomForestClassifier(n_estimators=100, max_depth=6, random_state=random_state)
    elif model_name == "RF Top-2 Features (Chicco & Jurman)":
        return RandomForestClassifier(n_estimators=100, max_depth=5, random_state=random_state)
    elif model_name == "Extra Trees (Ishaq et al. 2021)":
        return ExtraTreesClassifier(n_estimators=100, random_state=random_state)
    elif model_name == "Extra Trees + SMOTE (Ishaq et al. 2021)":
        return ExtraTreesClassifier(n_estimators=150, max_depth=10, random_state=random_state)
    elif model_name == "XGBoost Classifier":
        return xgb.XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, eval_metric='logloss', random_state=random_state)
    elif model_name == "LightGBM Classifier":
        return lgb.LGBMClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=random_state, verbose=-1)
    elif model_name == "MLP Neural Net (Umer et al. 2022)":
        return MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=random_state)
    elif model_name == "Proposed IoT Edge Lightweight Model":
        return ExtraTreesClassifier(
            n_estimators=120,
            max_depth=8,
            min_samples_split=3,
            min_samples_leaf=2,
            criterion='entropy',
            random_state=random_state
        )
    else:
        raise ValueError(f"Unknown model name: {model_name}")

def train_and_evaluate_all(data_dict, random_state=42):
    """
    Trains all candidate baseline and proposed models on train set and evaluates on test set.
    Includes SMOTE handling for class balancing.
    """
    X_train = data_dict['X_train']
    X_test = data_dict['X_test']
    X_train_scaled = data_dict['X_train_scaled']
    X_test_scaled = data_dict['X_test_scaled']
    y_train = data_dict['y_train']
    y_test = data_dict['y_test']

    # Generate SMOTE balanced dataset for SMOTE-enabled models
    smote = SMOTE(random_state=random_state)
    X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)

    models_to_run = [
        ("Logistic Regression", True, False, None),
        ("Random Forest (Chicco & Jurman 2020)", False, False, None),
        ("RF Top-2 Features (Chicco & Jurman)", False, False, ['serum_creatinine', 'ejection_fraction']),
        ("Extra Trees (Ishaq et al. 2021)", False, False, None),
        ("Extra Trees + SMOTE (Ishaq et al. 2021)", True, True, None),
        ("XGBoost Classifier", False, False, None),
        ("LightGBM Classifier", False, False, None),
        ("MLP Neural Net (Umer et al. 2022)", True, False, None),
        ("Proposed IoT Edge Lightweight Model", True, True, None)
    ]

    trained_models = {}
    results = {}

    from src.metrics import evaluate_classifier

    for name, use_scaled, use_smote, feature_subset in models_to_run:
        clf = get_model_pipeline(name, random_state=random_state)
        
        # Select appropriate dataset slice
        if feature_subset is not None:
            cur_X_train = X_train[feature_subset]
            cur_X_test = X_test[feature_subset]
        elif use_scaled:
            cur_X_train = X_train_scaled
            cur_X_test = X_test_scaled
        else:
            cur_X_train = X_train
            cur_X_test = X_test
            
        if use_smote:
            if feature_subset is not None:
                cur_X_tr, cur_y_tr = smote.fit_resample(cur_X_train, y_train)
            else:
                cur_X_tr, cur_y_tr = X_train_smote, y_train_smote
        else:
            cur_X_tr, cur_y_tr = cur_X_train, y_train

        # Fit model
        clf.fit(cur_X_tr, cur_y_tr)
        
        # Evaluate
        eval_res = evaluate_classifier(clf, cur_X_test, y_test)
        
        trained_models[name] = clf
        results[name] = eval_res

    return trained_models, results

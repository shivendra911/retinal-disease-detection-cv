"""
IRDAS Stage 1 — XGBoost Risk Triage Model
============================================

Clinical risk triage using tabular EHR data.
Prioritizes which diabetic patients should get retinal screening first.

Input: 6 clinical features (HbA1c, BP, age, diabetes duration, BMI, insulin)
Output: Risk score for having undetected diabetic retinopathy
Model: XGBoost with 5-fold stratified cross-validation

Target: AUC > 0.79 on NHANES test split

SHAP explainability shows which clinical factors drove each prediction.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import joblib
import os
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score


FEATURES = ['HbA1c_value', 'systolic_bp', 'age_years',
            'diabetes_duration_yrs', 'bmi', 'on_insulin_binary']
TARGET = 'has_referable_dr'  # binary: DR grade >= 2


def load_nhanes_data(data_dir):
    """
    Load pre-merged NHANES data.
    
    NHANES comes as multiple SAS transport files. Must be pre-merged into CSV.
    Merge: demographics (DEMO) + diabetes (DIQ) + labs (GHB) + BP (BPXO)
    
    Args:
        data_dir: Directory containing nhanes_merged.csv
    
    Returns:
        DataFrame with features and target
    """
    df = pd.read_csv(os.path.join(data_dir, 'nhanes_merged.csv'))
    df = df.dropna(subset=FEATURES + [TARGET])
    print(f"Dataset size: {len(df)} | DR positive rate: {df[TARGET].mean():.1%}")
    return df


def train_triage_model(df, save_dir='checkpoints'):
    """
    Train XGBoost triage model with 5-fold CV.
    
    Args:
        df: DataFrame with FEATURES and TARGET columns
        save_dir: Where to save the trained model
    
    Returns:
        model: Trained XGBClassifier
        explainer: SHAP TreeExplainer
        cv_aucs: List of 5-fold AUC values
    """
    X = df[FEATURES].values
    y = df[TARGET].values
    pos_ratio = (y == 0).sum() / max((y == 1).sum(), 1)
    
    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=pos_ratio,
        eval_metric='auc',
        random_state=42,
        use_label_encoder=False
    )
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_aucs = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        model.fit(
            X[train_idx], y[train_idx],
            eval_set=[(X[val_idx], y[val_idx])],
            verbose=False
        )
        val_pred = model.predict_proba(X[val_idx])[:, 1]
        auc = roc_auc_score(y[val_idx], val_pred)
        cv_aucs.append(auc)
        print(f"Fold {fold+1} AUC: {auc:.4f}")
    
    print(f"Mean CV AUC: {np.mean(cv_aucs):.4f} ± {np.std(cv_aucs):.4f}")
    
    # Final model on all data
    model.fit(X, y)
    os.makedirs(save_dir, exist_ok=True)
    joblib.dump(model, os.path.join(save_dir, 'triage_xgb.pkl'))
    
    # SHAP explainability
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    
    return model, explainer, cv_aucs


def rank_patients(patient_records_df, model):
    """
    Rank patients by predicted risk of undetected DR.
    
    Args:
        patient_records_df: DataFrame with clinical measurements
        model: Trained XGBoost model
    
    Returns:
        DataFrame sorted by risk score (highest risk first)
    """
    probs = model.predict_proba(patient_records_df[FEATURES].values)[:, 1]
    result = patient_records_df.copy()
    result['dr_risk_score'] = probs
    return result.sort_values('dr_risk_score', ascending=False)

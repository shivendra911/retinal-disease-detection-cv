import pandas as pd
import numpy as np
import os

def generate_mock_nhanes(n_samples=5000, output_dir='data'):
    """
    Generates a highly realistic synthetic NHANES-style dataset for diabetic retinopathy triage.
    """
    np.random.seed(42)
    os.makedirs(output_dir, exist_ok=True)
    
    # Base features
    age_years = np.random.normal(55, 15, n_samples)
    age_years = np.clip(age_years, 18, 90)
    
    # Diabetics tend to have higher BMI
    bmi = np.random.normal(28, 6, n_samples) + (age_years / 20)
    bmi = np.clip(bmi, 18, 50)
    
    # HbA1c is strongly correlated with diabetes duration and insulin usage
    diabetes_duration_yrs = np.where(np.random.rand(n_samples) > 0.3, 
                                     np.random.exponential(8, n_samples), 0)
    
    hba1c_value = np.random.normal(6.5, 1.5, n_samples) + (diabetes_duration_yrs * 0.15)
    hba1c_value = np.clip(hba1c_value, 4.0, 14.0)
    
    # Blood pressure
    systolic_bp = np.random.normal(120, 15, n_samples) + (age_years * 0.2) + (bmi * 0.3)
    
    # Insulin usage is highly probable for long duration or high HbA1c
    insulin_prob = 1 / (1 + np.exp(-(-5 + (hba1c_value * 0.5) + (diabetes_duration_yrs * 0.2))))
    on_insulin_binary = np.random.binomial(1, insulin_prob)
    
    # TARGET: has_referable_dr (DR grade >= 2)
    # DR is strongly caused by: high HbA1c, long diabetes duration, high BP
    risk_logit = -8 + (hba1c_value * 0.6) + (diabetes_duration_yrs * 0.2) + (systolic_bp * 0.02)
    dr_prob = 1 / (1 + np.exp(-risk_logit))
    has_referable_dr = np.random.binomial(1, dr_prob)
    
    df = pd.DataFrame({
        'age_years': age_years.astype(int),
        'bmi': np.round(bmi, 1),
        'diabetes_duration_yrs': np.round(diabetes_duration_yrs, 1),
        'HbA1c_value': np.round(hba1c_value, 1),
        'systolic_bp': systolic_bp.astype(int),
        'on_insulin_binary': on_insulin_binary,
        'has_referable_dr': has_referable_dr
    })
    
    out_path = os.path.join(output_dir, 'nhanes_merged.csv')
    df.to_csv(out_path, index=False)
    print(f"Generated synthetic NHANES dataset with {n_samples} records at {out_path}")
    print(f"DR prevalence: {df['has_referable_dr'].mean():.1%}")

if __name__ == "__main__":
    generate_mock_nhanes()

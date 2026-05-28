import os
import joblib
import pandas as pd
import numpy as np
import traceback

# --- Config ---
STAGE1_MODEL_PATH = os.path.join(os.path.dirname(__file__), '../checkpoints/triage_xgb.pkl')
STAGE2_MODEL_PATH = os.path.join(os.path.dirname(__file__), '../checkpoints/msdnet_best.pth')

# --- Global State ---
_triage_model = None

def load_triage_model():
    global _triage_model
    if _triage_model is None:
        try:
            _triage_model = joblib.load(STAGE1_MODEL_PATH)
            print("Loaded Stage 1 Triage Model successfully.")
        except Exception as e:
            print(f"Error loading Triage Model: {e}")
            _triage_model = None
    return _triage_model

def predict_triage_risk(patient_data: dict) -> dict:
    """
    Predicts the risk of DR for a single patient record.
    patient_data should be a dictionary mapping feature names to values.
    """
    model = load_triage_model()
    if model is None:
        return {"error": "Triage model not loaded or not found"}
        
    try:
        # Convert single dict to DataFrame
        df = pd.DataFrame([patient_data])
        
        # Ensure correct column order if possible, though XGBoost uses feature names
        risk_prob = model.predict_proba(df)[0][1] # Probability of Class 1 (High Risk)
        risk_score = float(risk_prob)
        
        return {
            "risk_score": risk_score,
            "risk_category": "High Risk" if risk_score > 0.5 else "Low Risk",
            "message": "Immediate screening recommended." if risk_score > 0.5 else "Routine annual screening."
        }
    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}

def validate_fundus_image(image_bytes: bytes) -> tuple[bool, str]:
    import cv2
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        return False, "Invalid image format."
        
    # Heuristic 1: Color distribution. Fundus images are highly vascular (red/orange).
    # In OpenCV BGR format, Red channel should strongly dominate Blue.
    b_mean = np.mean(img[:, :, 0])
    r_mean = np.mean(img[:, :, 2])
    
    # If blue is brighter than red, it is almost certainly a generic photograph/landscape, not a retina.
    if b_mean > r_mean:
        return False, "Image Rejected: Failed clinical color profile. The image does not appear to be a highly vascular retinal fundus scan."
        
    # Heuristic 2: Shape mask. Most fundus images have a dark circular FOV mask, 
    # leaving the corners totally black. We can check if the extreme corners are very dark.
    h, w = img.shape[:2]
    
    # Edge Case: Image resolution is too small for PyTorch convolutions
    if h < 224 or w < 224:
        return False, "Image Rejected: Resolution too low. Please upload an image of at least 224x224 pixels."
        
    corner_size = min(h, w) // 10
    top_left = img[0:corner_size, 0:corner_size]
    bottom_right = img[h-corner_size:h, w-corner_size:w]
    
    # If the corners are extremely bright (like a white background or sky), reject.
    corner_brightness = np.mean([np.mean(top_left), np.mean(bottom_right)])
    center_brightness = np.mean(img[h//2 - corner_size: h//2 + corner_size, w//2 - corner_size: w//2 + corner_size])
    
    # In a fundus image, the center is bright (optic disc/macula area) and corners are pitch black.
    # If corners are brighter than the center, it's a generic photo.
    if corner_brightness > center_brightness and corner_brightness > 100:
        return False, "Image Rejected: Failed structural profile. Missing the characteristic circular aperture mask of a fundus camera."

    return True, "Valid Fundus Image"


def predict_dr_image(image_bytes: bytes) -> dict:
    """
    Placeholder for Stage 2 PyTorch prediction.
    When the Kaggle training is done, we will load msdnet_best.pth here.
    """
    # 1. Input Validation
    is_valid, msg = validate_fundus_image(image_bytes)
    if not is_valid:
        return {
            "status": "rejected",
            "message": msg,
            "predicted_grade": "N/A"
        }

    # 2. PyTorch Inference (TODO)
    return {
        "status": "pending",
        "message": "Valid fundus detected! Stage 2 PyTorch inference will be activated once training completes.",
        "predicted_grade": "N/A"
    }

def generate_report(data: dict) -> dict:
    """
    Generates a patient-facing clinical report using the Stage 3 module,
    prepended with a professional clinical metric summary for the doctor.
    """
    import sys
    import os
    # Add project root to path to import stage3_comms
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if project_root not in sys.path:
        sys.path.append(project_root)
        
    try:
        from stage3_comms.multilingual_report import generate_patient_report
        
        # 1. Build Professional Clinical Summary
        dr_grade = data.get('dr_grade', 0)
        hr_present = data.get('hr_present', 0.0)
        language = data.get('language', 'english')
        
        prof_report = "CLINICAL METRICS SUMMARY (PROFESSIONAL)\n"
        prof_report += "========================================\n"
        prof_report += f"Predicted DR Grade: {dr_grade}/4\n"
        prof_report += f"Hypertensive Retinopathy Probability: {hr_present:.2f}\n"
        
        if data.get('Age') is not None:
            prof_report += f"\nTriage Inputs:\n"
            prof_report += f"- Age: {data.get('Age')}\n"
            prof_report += f"- BMI: {data.get('BMI')}\n"
            prof_report += f"- HbA1c: {data.get('HbA1c')}%\n"
            prof_report += f"- BP: {data.get('Systolic_BP')}/{data.get('Diastolic_BP')} mmHg\n"
            prof_report += f"- Cholesterol: {data.get('Cholesterol')} mg/dL\n"
            if data.get('Risk_Score') is not None:
                prof_report += f"- Triage Risk Score: {data.get('Risk_Score')*100:.1f}%\n"
                
        prof_report += "\n========================================\n"
        prof_report += "PATIENT COMMUNICATION (AI GENERATED)\n"
        prof_report += "========================================\n"
        
        # 2. Generate Patient Communication via LLM
        patient_text = generate_patient_report(
            dr_grade=dr_grade,
            hr_present=hr_present,
            dr_uncertainty=0.05,
            hr_uncertainty=0.05,
            heatmap_description="macula and optic disc",
            patient_language=language.lower()
        )
        
        final_report = prof_report + patient_text
        return {"status": "success", "report": final_report}
    except Exception as e:
        traceback.print_exc()
        return {"status": "error", "message": f"Failed to generate report: {str(e)}"}

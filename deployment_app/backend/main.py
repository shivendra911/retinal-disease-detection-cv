from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

from inference import predict_triage_risk, predict_dr_image, load_triage_model

app = FastAPI(title="IRDAS Backend API", version="1.0.0")

# Allow Frontend CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, restrict to frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize models on startup
@app.on_event("startup")
async def startup_event():
    print("Initializing models...")
    load_triage_model()

@app.get("/")
def read_root():
    return {"status": "ok", "message": "IRDAS Backend is running"}

class PatientData(BaseModel):
    # Biological constraints to prevent garbage data
    Age: float = Field(..., ge=0, le=120, description="Age in years")
    BMI: float = Field(..., ge=10, le=100, description="Body Mass Index")
    HbA1c: float = Field(..., ge=2, le=25, description="Hemoglobin A1c percentage")
    Systolic_BP: float = Field(..., ge=50, le=300, description="Systolic Blood Pressure")
    Diastolic_BP: float = Field(..., ge=30, le=200, description="Diastolic Blood Pressure")
    Cholesterol: float = Field(..., ge=50, le=600, description="Total Cholesterol")

@app.post("/api/triage")
def run_triage(data: PatientData):
    """
    Runs the Stage 1 Triage XGBoost model.
    """
    result = predict_triage_risk(data.dict())
    return result

MAX_FILE_SIZE = 15 * 1024 * 1024 # 15 MB
import asyncio

# Edge Case: GPU Out-Of-Memory Protection
# Prevent multiple users from triggering massive PyTorch inferences simultaneously
gpu_semaphore = asyncio.Semaphore(1)

@app.post("/api/predict")
async def run_predict(file: UploadFile = File(...)):
    """
    Runs the Stage 2 PyTorch model on an uploaded image.
    """
    # Edge Case: Not an image file
    if not file.content_type.startswith("image/"):
        return {"status": "rejected", "message": "Upload rejected: File must be an image format (JPEG, PNG, etc)."}
        
    image_bytes = await file.read()
    
    # Edge Case: Empty file or extremely small file
    if len(image_bytes) < 100:
        return {"status": "rejected", "message": "Upload rejected: File is empty or corrupted."}
        
    # Edge Case: Massively oversized file (DoS protection)
    if len(image_bytes) > MAX_FILE_SIZE:
        return {"status": "rejected", "message": f"Upload rejected: Image exceeds the 15MB limit."}
        
    # Safely lock the GPU to one inference at a time
    async with gpu_semaphore:
        result = predict_dr_image(image_bytes)
        
    return result

class ReportRequest(BaseModel):
    dr_grade: int = Field(default=0, ge=0, le=4, description="Diabetic Retinopathy Grade (0-4)")
    hr_present: float = Field(default=0.0, ge=0.0, le=1.0, description="Probability of Hypertensive Retinopathy")
    language: str = Field(default="english", description="Target language for the report")
    # Clinical Metrics
    Age: float = Field(default=None)
    BMI: float = Field(default=None)
    HbA1c: float = Field(default=None)
    Systolic_BP: float = Field(default=None)
    Diastolic_BP: float = Field(default=None)
    Cholesterol: float = Field(default=None)
    Risk_Score: float = Field(default=None)

@app.post("/api/report")
def run_report(data: ReportRequest):
    """
    Generates a patient-facing clinical report using the Stage 3 LLM module.
    """
    from inference import generate_report
    result = generate_report(data.dict())
    return result

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.0", port=8000, reload=True)

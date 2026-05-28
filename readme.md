# IRDAS — Integrated Retinal Disease Analysis System

An AI-powered, three-stage retinal disease screening pipeline designed for low-resource clinical deployment in India. 

![IRDAS Architecture overview](https://img.shields.io/badge/Architecture-Three_Stage_Pipeline-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-red)
![FastAPI](https://img.shields.io/badge/FastAPI-Production-green)
![React](https://img.shields.io/badge/React-19-blue)

## 📌 Overview

**IRDAS** (Integrated Retinal Disease Analysis System) is a comprehensive healthcare ML project for computer vision. This system is tailored to assist in the early detection and management of retinal diseases—specifically **Diabetic Retinopathy (DR)** and **Hypertensive Retinopathy (HR)**—with an emphasis on low-resource environments. 

The core contribution is a novel computer vision architecture, the **Multi-Scale Disease Network (MSDNet)**, which simultaneously predicts multiple disease severities and captures microvascular features crucial for accurate diagnosis.

## 🚀 Repository Structure

The project correctly isolates the production deployment configuration from the historical research and development archive.

```bash
IRDAS/
│
├── 🚀 deployment_app/               # Production-ready codebase
│   ├── backend/                     # FastAPI inference API 
│   │   ├── main.py                  # Endpoints & Pydantic models
│   │   ├── inference.py             # Inference model wrapper
│   │   ├── utils.py                 # Core preprocessing funcs
│   │   └── requirements.txt         # Minimal dependency list
│   │
│   ├── frontend/                    # Vite + React 19 Frontend
│   │   ├── src/
│   │   └── package.json
│   │
│   └── models_cache/                # Where the trained .pth weights live
│
└── 🔬 research_archive/             # Historical Research, Models & Training
    ├── analysis_logs/               # Diagnostic outputs & html reports
    ├── documentation_and_guides/    # Development action plans & design docs
    ├── finalarchitecture/           # Core PyTorch model architectures
    ├── notebooks/                   # Core Kaggle experiments & dev ipynbs
    ├── papers_and_reports/          # LaTeX Journal drafts & architectures
    ├── scripts_and_misc/            # Development training python scripts
    └── (other research subfolders: config, evaluation, xai, losses, etc.)
```

## 🛠️ Installation & Setup (Deployment App)

To run the streamlined system for inference:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/USERNAME/retinal-disease-detection-cv.git
   cd retinal-disease-detection-cv
   ```

2. **Run Backend (FastAPI):**
   ```bash
   cd deployment_app/backend
   pip install -r requirements.txt
   # Download the 'final_irdas.pth' and place it into deployment_app/models_cache
   uvicorn main:app --reload
   ```

3. **Run Frontend (React):**
   ```bash
   cd deployment_app/frontend
   npm install
   npm run dev
   ```

## 🧠 The Three-Stage Pipeline

IRDAS uses a robust strategy originally researched in `research_archive/`:
1. **Risk Triage (Pre-Screening):** XGBoost classification using EHR.
2. **MSDNet (CV Core):** PyTorch EfficientNet-B0 backbone with FPN.
3. **Communication & Reporting:** Gemini API via LangChain for plain-language multi-lingual reports.
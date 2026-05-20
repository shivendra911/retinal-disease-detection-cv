# IRDAS — Integrated Retinal Disease Analysis System

An AI-powered, three-stage retinal disease screening pipeline designed for low-resource clinical deployment in India. 

![IRDAS Architecture overview](https://img.shields.io/badge/Architecture-Three_Stage_Pipeline-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-red)
![Status](https://img.shields.io/badge/Status-Research_Phase-orange)

## 📌 Overview

**IRDAS** (Integrated Retinal Disease Analysis System) is a comprehensive healthcare ML research project for computer vision. This system is tailored to assist in the early detection and management of retinal diseases—specifically **Diabetic Retinopathy (DR)** and **Hypertensive Retinopathy (HR)**—with an emphasis on low-resource environments. 

The core contribution is a novel computer vision architecture, the **Multi-Scale Disease Network (MSDNet)**, which simultaneously predicts multiple disease severities and captures microvascular features crucial for accurate diagnosis.

## 🚀 The Three-Stage Pipeline

IRDAS is broken down into three logical stages mimicking a real-world clinical screening pipeline:

### Stage 1: Risk Triage (Pre-Screening)
- **Input:** Clinical Electronic Health Records (EHR) data (e.g., age, HbA1c, blood pressure).
- **Model:** XGBoost.
- **Output:** Patient priority queue, optimizing resource allocation by determining which patients need immediate imaging and screening.

### Stage 2: MSDNet (CV Core)
- **Input:** Retinal fundus images.
- **Model Architecture:** 
  - **Backbone:** EfficientNet-B0 (for edge-friendly efficiency).
  - **Feature Pyramid Network (FPN):** To capture microaneurysms at varied scales.
  - **Disease-specific Branches:** CBAM attention modules for Diabetic and Hypertensive Retinopathy.
  - **Vessel Decoder:** Multi-task learning for vessel segmentation to force the model to learn vascular structures.
  - **Uncertainty Estimation:** MC Dropout for epistemic uncertainty.
  - **Contrastive Learning:** Disentanglement loss to isolate DR and HR features.
- **Output:** DR severity (grades 0-4), HR presence probability, spatial heatmaps, and a model uncertainty score.

### Stage 3: Communication & Reporting
- **Input:** Multi-disease predictions, uncertainty scores, and XAI explanations from Stage 2.
- **Model:** Gemini API via LangChain.
- **Output:** Plain-language, multilingual patient reports generated in various Indic languages for enhanced patient accessibility.

## 🧠 Explainable AI (XAI)

To build trust with clinicians, IRDAS heavily relies on **Grad-CAM++** to generate high-resolution disease-specific spatial heatmaps. These heatmaps highlight the specific retinal regions (like exudates, hemorrhages, or microaneurysms) that influenced the model's predictions.

## 📂 Datasets

The project is trained, evaluated, and cross-validated across multiple public and clinical datasets to ensure robust generalization:
- **APTOS 2019 Blindness Detection:** Primary dataset for Diabetic Retinopathy classification.
- **HRDC 2023:** For Hypertensive Retinopathy classification.
- **DRIVE:** Ground truth for blood vessel segmentation.
- **IDRiD:** Used as a holdout test set to measure generalization capability.
- **NHANES:** Clinical and demographic data for training the Stage 1 Risk Triage model.

## 📊 Target Metrics

The system aims for high clinical reliability, tracking the following key performance indicators:
- **DR QWK (APTOS val):** > 0.89
- **HR AUC (HRDC val):** > 0.91
- **Vessel Dice (DRIVE test):** > 0.76
- **ECE (Model Calibration):** < 0.06

## ⚙️ Tech Stack & Tools

- **Deep Learning / CV:** PyTorch, TorchVision, timm, albumentations, OpenCV, SMP (Segmentation Models PyTorch)
- **Tabular ML:** XGBoost, Scikit-Learn
- **Explainable AI:** Grad-CAM++ (Captum / pytorch-grad-cam)
- **LLM / GenAI:** LangChain, Google Gemini API
- **Experiment Tracking:** Weights & Biases (WandB)
- **Data & Config:** Pandas, NumPy, PyYAML

## 📁 Repository Structure

```bash
IRDAS/
│
├── checkpoints/             # Trained model weights (.pth files)
├── config/                  # YAML configurations (hyperparameters, paths)
├── data/                    # Dataset loaders and PyTorch Dataset classes
│   ├── aptos_dataset.py
│   ├── drive_dataset.py
│   ├── hrdc_dataset.py
│   └── idrid_dataset.py
├── evaluation/              # Metrics calculation (QWK, AUC, Dice, ECE)
├── experiments/             # Ablation study configs (Baseline, FPN, Vessel, etc.)
├── losses/                  # Custom objective functions (Task, Vessel, Disentanglement)
├── models/                  # PyTorch model definitions (MSDNet, Backbone, FPN, CBAM)
├── notebooks/               # Kaggle training scripts and educational guides
│   ├── kaggle_part1_setup.py          # Environment and dataset preparation
│   ├── kaggle_part2_training.py       # Main model training loop
│   ├── kaggle_part3_eval.py           # Evaluation pipeline and metrics
│   ├── ModelTrainingEndToEnd_Part1.py # Educational guide on foundations & prep
│   ├── ModelTrainingEndToEnd_Part2.py # Educational guide on architecture & training
│   └── ModelTrainingEndToEnd_Part3.py # Educational guide on XAI & reporting
├── outputs/                 # XAI samples and preprocessing visualizations
├── preprocessing/           # Fundus image enhancement (CLAHE, OD Suppression)
├── stage1_triage/           # EHR-based XGBoost triage model
├── stage3_comms/            # Multilingual LLM-based report generation
├── xai/                     # Explainability tools (Grad-CAM++)
├── train.py                 # Main training orchestrator script
└── requirements.txt         # Pip dependency constraints
```

## 🛠️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/USERNAME/retinal-disease-detection-cv.git
   cd retinal-disease-detection-cv
   ```

2. **Create a Conda environment:**
   ```bash
   conda create -n irdas python=3.10 -y
   conda activate irdas
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

*Note: For local development, CPU-version PyTorch is recommended unless you have a dedicated local GPU. All heavy training is designed to run via Kaggle Notebooks using NVIDIA T4 GPUs.*
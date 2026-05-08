# RetinaVisionAI

AI-powered retinal disease diagnosis system using retinal fundus image analysis and transfer learning.

---

## 📌 Overview

RetinaVisionAI is a deep learning-based healthcare project designed to assist in the early detection of retinal diseases such as:

- Diabetic Retinopathy
- Glaucoma
- Retinal abnormalities

The system analyzes retinal fundus images using pretrained convolutional neural networks (CNNs) and provides automated predictions with confidence scores.

This project aims to support ophthalmologists and improve accessibility to retinal screening in low-resource environments.

---

## 🚀 Features

- Retinal image upload
- AI-based disease prediction
- Transfer learning using pretrained CNN models
- Grad-CAM heatmap visualization
- Confidence score generation
- User-friendly interface
- Medical report generation

---

## 🧠 AI Model

The project uses transfer learning with pretrained deep learning models such as:

- MobileNetV2
- EfficientNet
- ResNet50

These models are fine-tuned for retinal disease classification.

---

## 📂 Dataset

Datasets used:

- DIARETDB1
- APTOS 2019 Blindness Detection
- EyePACS
- MESSIDOR

---

## ⚙️ Tech Stack

### Frontend
- Streamlit / React

### Backend
- Flask

### AI/ML
- TensorFlow
- Keras
- OpenCV
- NumPy
- Pandas

---

## 🏗️ System Workflow

Retinal Image Upload  
↓  
Image Preprocessing  
↓  
Feature Extraction using CNN  
↓  
Disease Prediction  
↓  
Heatmap Visualization  
↓  
Result & Report Generation

---

## 📊 Model Capabilities

The system can:
- Detect retinal abnormalities
- Classify diseased vs healthy retinal images
- Highlight affected retinal regions
- Provide confidence-based predictions

---

## 🔍 Explainable AI

Grad-CAM visualization is used to improve interpretability by highlighting retinal regions influencing the model prediction.

---

## 📁 Project Structure

```bash
RetinaVisionAI/
│
├── app/
├── models/
├── static/
├── templates/
├── dataset/
├── notebooks/
├── app.py
├── requirements.txt
└── README.md
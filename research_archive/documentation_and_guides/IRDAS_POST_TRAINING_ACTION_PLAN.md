# IRDAS: Post-Training Action Plan

This guide breaks down exactly what to do next now that your core computer vision model (MSDNet - Stage 2) is trained. You will now focus on evaluating the model, building the Stage 1 triage filter, adding Stage 3 AI communications, and ultimately stitching them together.

Follow these parts sequentially.

---

## Part 1: Model Evaluation & XAI Generation (Phase 5 & 9)
*Now that the model has learned, we need to prove it works and interpret its decisions.*

- [ ] **Step 1.1: Download Checkpoints**
  - Download `msdnet_best.pth` from your Kaggle output.
  - Place it in your local `checkpoints/` directory.
- [ ] **Step 1.2: Run Evaluation Metrics**
  - Open `notebooks/kaggle_part3_eval.py` or write a local evaluation script.
  - Run the validation set through the model to calculate:
    - **Diabetic Retinopathy (DR):** Quadratic Weighted Kappa (QWK). Target > 0.89.
    - **Hypertensive Retinopathy (HR):** AUC. Target > 0.91.
    - **Generalization:** Test on the IDRiD dataset to see if it holds up.
- [ ] **Step 1.3: Generate Grad-CAM++ Heatmaps**
  - Implement/Run `xai/gradcam_branches.py`.
  - Pass 5-10 varied fundus images through the model.
  - Save the output heatmaps in `outputs/xai_examples/`. You will need these images to write your research paper.

---

## Part 2: Stage 1 - Risk Triage Model (Phase 7)
*Build the tabular (CPU-bound) model to decide who needs screening based on health records.*

- [ ] **Step 2.1: Data Preparation**
  - Prepare the NHANES dataset (or mock tabular EHR data) containing features like: `Age`, `HbA1c`, `Systolic/Diastolic BP`, `Years with Diabetes`.
- [ ] **Step 2.2: Train XGBoost Model**
  - Edit `stage1_triage/triage_model.py`.
  - Set up an XGBoost Classifier training loop using Scikit-Learn pipelines.
  - Target variable: `Needs_Screening (1 or 0)`.
- [ ] **Step 2.3: Evaluate and Save**
  - Calculate ROC-AUC for the Triage model (Target > 0.79).
  - Save the model weights locally as `checkpoints/triage_xgb.pkl`.

---

## Part 3: Stage 3 - Multilingual LLM Communications (Phase 8)
*Translate raw model outputs into human-readable clinical reports.*

- [ ] **Step 3.1: API Setup**
  - Get your free Google Gemini API key from Google AI Studio.
  - Store it locally in a `.env` file (`GEMINI_API_KEY=your_key_here`). Make sure `.env` is NOT committed to GitHub.
- [ ] **Step 3.2: LangChain Integration**
  - Edit `stage3_comms/multilingual_report.py`.
  - Create a LangChain prompt template. It should accept variables: `patient_name`, `triage_score`, `dr_grade`, `hr_prob`, `uncertainty`.
- [ ] **Step 3.3: Multilingual Generation**
  - Configure the LLM to output a structured clinical report in English, plus a localized translation (e.g., Hindi or regional languages) intended for the patient.
  - Test the prompt to ensure it doesn't give dangerous medical advice (add a medical disclaimer).

---

## Part 4: API Assembly & Deployment Prep (Phase 11)
*Stitch all 3 stages together into a single application.*

- [ ] **Step 4.1: Build the Flask API**
  - Create `app.py` in the root (refer to Section 8 of `IRDAS_PROJECT_COMPLETION_GUIDE.md`).
  - Create a `/predict` POST endpoint.
  - The endpoint should:
    1. Accept patient tabular data + fundus image.
    2. Run Stage 1 (XGBoost).
    3. Run Stage 2 (MSDNet inference).
    4. Run Stage 3 (Gemini report generation).
    5. Return a complete JSON payload.
- [ ] **Step 4.2: Build Docker Container**
  - Write a `Dockerfile` and `requirements_deploy.txt`.
  - Test the container locally (`docker build` and `docker run`).
- [ ] **Step 4.3: Deploy to AWS**
  - Use your AWS Student Pack credits.
  - Spin up a basic `t3.medium` EC2 instance, copy your deployed container over, and expose port 5000.

---

## Part 5: Research Paper & Wrap-up (Phase 10)
*Document your findings.*

- [ ] **Step 5.1: Compile Results**
  - Gather all WandB loss curves, metrics (from Part 1), and Heatmaps (from Part 1).
- [ ] **Step 5.2: Ablation Study Writeup**
  - Use your ablation study results to prove *why* you chose FPN and CBAM.
- [ ] **Step 5.3: Status Update**
  - Open `IRDAS_ENHANCED_PLAN.md` and check off `[x]` for all completed phases!
# Assignment 3: Medical Condition Prediction (End-to-End)

This project implements a full pipeline for predicting medical conditions from patient-reported symptoms.

## What is included

- `main.ipynb`: fully executed notebook with:
  - paradigm selection (Modeling-Inference-Learning)
  - preprocessing and feature engineering explanation
  - diagram outputs in code sections
  - comparison of **8 algorithms**
  - final model selection and artifact export
  - deployment flow diagram
  - data storytelling for next training cycle
- `data/generated/`: generated large CSV dataset (1,000,000 rows split into 4 files)
- `artifacts/`: trained best model + encoder + metrics
- `api/`: FastAPI backend for inference
- `frontend/`: React + TypeScript + Radix + Tailwind web app

## Data source and generated output

Raw source used:
- `https://raw.githubusercontent.com/anujdutt9/Disease-Prediction-from-Symptoms/master/dataset/training_data.csv`
- `https://raw.githubusercontent.com/anujdutt9/Disease-Prediction-from-Symptoms/master/dataset/test_data.csv`

Generated outputs:
- `data/generated/symptom_disease_1m_part_01.csv`
- `data/generated/symptom_disease_1m_part_02.csv`
- `data/generated/symptom_disease_1m_part_03.csv`
- `data/generated/symptom_disease_1m_part_04.csv`

Total generated rows: **1,000,000**.

## Reproducible commands

### 1) Generate large CSV dataset

```bash
python3 scripts/generate_large_dataset.py
```

### 2) Train and compare models (8 algorithms)

```bash
python3 scripts/train_models.py
```

This produces:
- `artifacts/model_comparison.csv`
- `artifacts/best_model.joblib`
- `artifacts/label_encoder.joblib`
- `artifacts/model_metadata.json`

### 3) Run API

```bash
pip install -r api/requirements.txt
uvicorn api.main:app --reload --port 8000
```

API endpoints:
- `GET /health`
- `GET /symptoms`
- `POST /predict`

### 4) Run web app

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`.

## Architecture

User selects symptoms in React UI -> request to FastAPI `/predict` -> model artifact inference -> top disease predictions returned to UI.

## Important note

This is an educational prototype and **not** a clinical diagnosis system.

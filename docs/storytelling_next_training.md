# Storytelling for Next Model Training

We started with symptom-only binary signals. That helps classify obvious disease patterns, but hospital-grade prediction needs context.

For the next training iteration, collect:

1. Demographics (`age`, `sex`, region)
2. Vitals (`SpO2`, blood pressure, heart rate, temperature)
3. Lab panels (`ALT/AST`, bilirubin, CBC, CRP, creatinine)
4. Timeline (symptom onset date, duration, progression)
5. Comorbidity history (diabetes, hypertension, asthma, kidney disease)
6. Outcome labels (confirmed diagnosis, treatment response, recovery time)

Narrative:
A patient with cough + fatigue could be viral infection, lung condition, anemia, or cardiac issue. Symptoms alone are ambiguous. Once we add SpO2, CBC, CRP, and comorbidity history, uncertainty drops and model confidence becomes clinically more useful.

Data quality priorities:
- Standardize symptom vocabulary with one coding scheme.
- Keep timestamps for every measurement.
- Track missingness reason (not measured vs unknown).
- Retain clinician-confirmed diagnosis for ground truth.

Result:
The next model should shift from symptom-only screening toward richer differential diagnosis and better treatment-support recommendations.

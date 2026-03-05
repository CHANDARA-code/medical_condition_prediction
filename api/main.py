from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import List

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

def _resolve_artifacts_dir() -> Path:
    env_dir = os.getenv("MODEL_ARTIFACTS_DIR", "").strip()
    candidates: list[Path] = []

    if env_dir:
        candidates.append(Path(env_dir))

    candidates.extend(
        [
            Path(__file__).resolve().parents[1] / "artifacts",
            Path(__file__).resolve().parent / "artifacts",
            Path.cwd() / "artifacts",
        ]
    )

    for candidate in candidates:
        model_file = candidate / "best_model.joblib"
        meta_file = candidate / "model_metadata.json"
        if model_file.exists() and meta_file.exists():
            return candidate

    searched = ", ".join(str(p) for p in candidates)
    raise RuntimeError(
        "Model artifacts not found. Expected `best_model.joblib` and `model_metadata.json`. "
        f"Searched: {searched}. Set MODEL_ARTIFACTS_DIR if needed."
    )


ARTIFACTS_DIR = _resolve_artifacts_dir()
model_path = ARTIFACTS_DIR / "best_model.joblib"
metadata_path = ARTIFACTS_DIR / "model_metadata.json"

model = joblib.load(model_path)
metadata = json.loads(metadata_path.read_text())
features: list[str] = metadata["features"]
classes: list[str] = metadata["classes"]

SUPPORTED_LOCALES = ("en", "km")

MESSAGES = {
    "en": {
        "at_least_one_symptom": "Please provide at least one symptom",
        "unknown_symptoms_prefix": "Unknown symptoms",
    },
    "km": {
        "at_least_one_symptom": "សូមជ្រើសរោគសញ្ញាយ៉ាងហោចណាស់មួយ",
        "unknown_symptoms_prefix": "រកមិនឃើញរោគសញ្ញា",
    },
}

SYMPTOM_KM_OVERRIDES = {
    "spotting urination": "មានឈាមតិចៗក្នុងទឹកនោម",
    "foul smell of urine": "ទឹកនោមមានក្លិនខ្លាំង",
    "continuous feel of urine": "មានអារម្មណ៍ចង់នោមជាប់ៗ",
    "dischromic patches": "ស្នាមពណ៌ស្បែកមិនស្មើ",
    "fluid overload 1": "រាងកាយផ្ទុករាវលើស (1)",
    "toxic look (typhos)": "សភាពរាងកាយពុល (Typhos)",
    "small dents in nails": "ក្រចកមានរន្ធតូចៗ",
    "red sore around nose": "ដំបៅក្រហមជុំវិញច្រមុះ",
}

SYMPTOM_TOKEN_KM_TRANSLATIONS = {
    "itching": "រមាស់",
    "skin": "ស្បែក",
    "rash": "កន្ទួល",
    "nodal": "កូនកណ្តុរ",
    "eruptions": "ផ្ទុះឡើង",
    "continuous": "បន្តបន្ទាប់",
    "sneezing": "កណ្តាស់",
    "shivering": "ញ័រ",
    "chills": "ញាក់",
    "joint": "សន្លាក់",
    "pain": "ឈឺ",
    "stomach": "ក្រពះ",
    "acidity": "អាស៊ីតខ្ពស់",
    "ulcers": "ដំបៅ",
    "on": "លើ",
    "tongue": "អណ្តាត",
    "muscle": "សាច់ដុំ",
    "wasting": "ស្តើង",
    "vomiting": "ក្អួត",
    "burning": "រលាក",
    "micturition": "នោម",
    "spotting": "ស្នាមឈាម",
    "urination": "ទឹកនោម",
    "fatigue": "អស់កម្លាំង",
    "weight": "ទម្ងន់",
    "gain": "ឡើង",
    "anxiety": "បារម្ភ",
    "cold": "ត្រជាក់",
    "hands": "ដៃ",
    "and": "និង",
    "feets": "ជើង",
    "mood": "អារម្មណ៍",
    "swings": "ប្រែប្រួល",
    "loss": "បាត់បង់",
    "restlessness": "មិនស្ងប់",
    "lethargy": "សន្លឹម",
    "patches": "ស្នាម",
    "in": "ក្នុង",
    "throat": "បំពង់ក",
    "irregular": "មិនទៀងទាត់",
    "sugar": "ស្ករ",
    "level": "កម្រិត",
    "cough": "ក្អក",
    "high": "ខ្ពស់",
    "fever": "គ្រុន",
    "sunken": "លង់",
    "eyes": "ភ្នែក",
    "breathlessness": "ដង្ហើមខ្លី",
    "sweating": "ញើសច្រើន",
    "dehydration": "ខ្វះជាតិទឹក",
    "indigestion": "អាហារមិនរលាយ",
    "headache": "ឈឺក្បាល",
    "yellowish": "លឿង",
    "dark": "ងងឹត",
    "urine": "ទឹកនោម",
    "nausea": "ចង់ក្អួត",
    "of": "នៃ",
    "appetite": "ចំណង់អាហារ",
    "behind": "ខាងក្រោយ",
    "the": "",
    "back": "ខ្នង",
    "constipation": "ទល់លាមក",
    "abdominal": "ពោះ",
    "diarrhoea": "រាគ",
    "mild": "ស្រាល",
    "yellow": "លឿង",
    "yellowing": "លឿង",
    "acute": "ស្រួចស្រាវ",
    "liver": "ថ្លើម",
    "failure": "ខ្សោយ",
    "fluid": "រាវ",
    "overload": "លើស",
    "swelling": "ហើម",
    "swelled": "ហើម",
    "lymph": "កូនកណ្តុរ",
    "nodes": "ក្រពេញ",
    "malaise": "មិនស្រួលខ្លួន",
    "blurred": "ព្រិល",
    "distorted": "ខូចទ្រង់ទ្រាយ",
    "vision": "ចក្ខុវិស័យ",
    "phlegm": "ស្លេស",
    "irritation": "រលាក",
    "redness": "ក្រហម",
    "sinus": "ប្រហោងច្រមុះ",
    "pressure": "សម្ពាធ",
    "runny": "ហូរ",
    "nose": "ច្រមុះ",
    "congestion": "តឹងច្រមុះ",
    "chest": "ទ្រូង",
    "weakness": "ខ្សោយ",
    "limbs": "អវយវៈ",
    "fast": "លឿន",
    "heart": "បេះដូង",
    "rate": "ចង្វាក់",
    "during": "ពេល",
    "bowel": "ពោះវៀន",
    "movements": "ចលនា",
    "anal": "រន្ធគូថ",
    "region": "តំបន់",
    "bloody": "មានឈាម",
    "stool": "លាមក",
    "anus": "រន្ធគូថ",
    "neck": "ក",
    "dizziness": "វិលមុខ",
    "cramps": "រមួលក្រពើ",
    "bruising": "ជាំ",
    "obesity": "ធាត់លើស",
    "swollen": "ហើម",
    "legs": "ជើង",
    "blood": "ឈាម",
    "vessels": "សរសៃឈាម",
    "puffy": "ហើមពពុះ",
    "face": "មុខ",
    "enlarged": "រីកធំ",
    "thyroid": "ទីរ៉ូអ៊ីត",
    "brittle": "ផុយ",
    "nails": "ក្រចក",
    "extremeties": "ចុងដៃជើង",
    "excessive": "លើសកម្រិត",
    "hunger": "ឃ្លាន",
    "extra": "ក្រៅ",
    "marital": "អាពាហ៍ពិពាហ៍",
    "contacts": "ទំនាក់ទំនង",
    "drying": "ស្ងួត",
    "tingling": "ស្រួចស្រាវ",
    "lips": "បបូរមាត់",
    "slurred": "និយាយមិនច្បាស់",
    "speech": "សុន្ទរកថា",
    "knee": "ជង្គង់",
    "hip": "ត្រគាក",
    "stiff": "រឹង",
    "joints": "សន្លាក់",
    "movement": "ចលនា",
    "stiffness": "រឹងតឹង",
    "spinning": "វិល",
    "balance": "តុល្យភាព",
    "unsteadiness": "មិនស្ថិរភាព",
    "one": "មួយ",
    "body": "រាងកាយ",
    "side": "ម្ខាង",
    "smell": "ក្លិន",
    "bladder": "ប្លោកនោម",
    "discomfort": "មិនស្រួល",
    "foul": "ស្អុយ",
    "smell": "ក្លិន",
    "feel": "អារម្មណ៍",
    "passage": "បញ្ចេញ",
    "gases": "ឧស្ម័ន",
    "internal": "ខាងក្នុង",
    "toxic": "ពុល",
    "look": "រូបរាង",
    "typhos": "ទីហ្វូស",
    "depression": "ធ្លាក់ទឹកចិត្ត",
    "irritability": "ងាយខឹង",
    "altered": "ផ្លាស់ប្តូរ",
    "sensorium": "ស្មារតី",
    "red": "ក្រហម",
    "spots": "ចំណុច",
    "over": "លើ",
    "belly": "ពោះ",
    "abnormal": "មិនប្រក្រតី",
    "menstruation": "មករដូវ",
    "dischromic": "ពណ៌មិនស្មើ",
    "watering": "ហូរទឹក",
    "from": "ពី",
    "increased": "កើនឡើង",
    "polyuria": "នោមញឹក",
    "family": "គ្រួសារ",
    "history": "ប្រវត្តិ",
    "mucoid": "ស្លេសខាប់",
    "sputum": "កំហាក",
    "rusty": "ពណ៌ច្រេះ",
    "lack": "ខ្វះ",
    "concentration": "ការផ្តោតអារម្មណ៍",
    "visual": "ចក្ខុ",
    "disturbances": "រំខាន",
    "receiving": "បានទទួល",
    "transfusion": "បញ្ចូលឈាម",
    "unsterile": "មិនស្អាត",
    "injections": "ចាក់ថ្នាំ",
    "coma": "សន្លប់",
    "bleeding": "ហូរឈាម",
    "distention": "ពោះហើម",
    "abdomen": "ពោះ",
    "alcohol": "ស្រា",
    "consumption": "ការប្រើប្រាស់",
    "prominent": "លេចចេញ",
    "veins": "សរសៃឈាមវ៉ែន",
    "calf": "កំភួនជើង",
    "palpitations": "បេះដូងលោតញាប់",
    "painful": "ឈឺចាប់",
    "walking": "ដើរ",
    "pus": "ខ្ទុះ",
    "filled": "ពេញ",
    "pimples": "មុនពពុះ",
    "blackheads": "មុនក្បាលខ្មៅ",
    "scurring": "ស្បែករដុប",
    "peeling": "របក",
    "silver": "ប្រាក់",
    "like": "ដូច",
    "dusting": "ធូលី",
    "small": "តូច",
    "dents": "រន្ធ",
    "inflammatory": "រលាក",
    "blister": "ពពុះទឹក",
    "sore": "ដំបៅ",
    "around": "ជុំវិញ",
    "crust": "សំបក",
    "ooze": "ហូររាវ",
}

CONDITION_KM_TRANSLATIONS = {
    "(vertigo) paroymsal positional vertigo": "វឺទីហ្គោប្រភេទទីតាំងឆ្លាស់",
    "aids": "អេដស៍",
    "acne": "មុន",
    "alcoholic hepatitis": "រលាកថ្លើមដោយសារស្រា",
    "allergy": "អាលែកស៊ី",
    "arthritis": "រលាកសន្លាក់",
    "bronchial asthma": "ហឺតទងសួត",
    "cervical spondylosis": "រលាកឆ្អឹងកខ្នងផ្នែកក",
    "chicken pox": "អុតស្វាយ",
    "chronic cholestasis": "ទឹកប្រមាត់ជាប់រ៉ាំរ៉ៃ",
    "common cold": "ផ្តាសាយធម្មតា",
    "dengue": "គ្រុនឈាម",
    "diabetes": "ទឹកនោមផ្អែម",
    "dimorphic hemmorhoids(piles)": "ឬសដូងបាត",
    "drug reaction": "ប្រតិកម្មថ្នាំ",
    "fungal infection": "ឆ្លងផ្សិត",
    "gerd": "ជំងឺអាស៊ីតក្រពះត្រឡប់",
    "gastroenteritis": "រលាកក្រពះពោះវៀន",
    "heart attack": "គាំងបេះដូង",
    "hepatitis b": "រលាកថ្លើម B",
    "hepatitis c": "រលាកថ្លើម C",
    "hepatitis d": "រលាកថ្លើម D",
    "hepatitis e": "រលាកថ្លើម E",
    "hypertension": "លើសឈាម",
    "hyperthyroidism": "ទីរ៉ូអ៊ីតសកម្មលើស",
    "hypoglycemia": "ជាតិស្ករទាប",
    "hypothyroidism": "ទីរ៉ូអ៊ីតសកម្មទាប",
    "impetigo": "ជំងឺស្បែក Impetigo",
    "jaundice": "លឿង",
    "malaria": "គ្រុនចាញ់",
    "migraine": "ឈឺក្បាលប្រកាំង",
    "osteoarthristis": "រលាកសន្លាក់ឆ្អឹង",
    "paralysis (brain hemorrhage)": "អសកម្មរាងកាយ (ឈាមហូរក្នុងខួរក្បាល)",
    "peptic ulcer diseae": "ដំបៅក្រពះ",
    "pneumonia": "រលាកសួត",
    "psoriasis": "សូរ៉ាយអាស៊ីស",
    "tuberculosis": "របេង",
    "typhoid": "ទីហ្វូអ៊ីត",
    "urinary tract infection": "ឆ្លងផ្លូវទឹកនោម",
    "varicose veins": "សរសៃឈាមវ៉ារីកូស",
    "hepatitis a": "រលាកថ្លើម A",
}


def _normalize_space(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _normalize_locale(lang: str | None) -> str:
    if not lang:
        return "en"
    value = lang.strip().lower().replace("_", "-")
    if value.startswith("km"):
        return "km"
    return "en"


def _message(key: str, locale: str) -> str:
    return MESSAGES.get(locale, MESSAGES["en"])[key]


def _humanize_code(code: str) -> str:
    raw = _normalize_space(code.replace("_", " ").replace(".", " "))
    acronyms = {"gerd", "hiv", "aids"}
    words = [w.upper() if w.lower() in acronyms else w.capitalize() for w in raw.split(" ")]
    return " ".join(words)


def _localize_symptom_label(code: str, locale: str) -> str:
    english = _humanize_code(code)
    if locale == "en":
        return english

    normalized = _normalize_space(code.replace("_", " ").replace(".", " ")).lower()
    if normalized in SYMPTOM_KM_OVERRIDES:
        return SYMPTOM_KM_OVERRIDES[normalized]

    translated_words = []
    for word in normalized.split(" "):
        translated = SYMPTOM_TOKEN_KM_TRANSLATIONS.get(word, word)
        if translated:
            translated_words.append(translated)
    return " ".join(translated_words) if translated_words else english


def _localize_condition_label(condition: str, locale: str) -> str:
    normalized_condition = _normalize_space(condition)
    if locale == "en":
        return normalized_condition
    key = normalized_condition.lower()
    return CONDITION_KM_TRANSLATIONS.get(key, normalized_condition)


def _symptom_payload(code: str, locale: str) -> dict[str, str]:
    label_en = _localize_symptom_label(code, "en")
    label_km = _localize_symptom_label(code, "km")
    label = label_en if locale == "en" else label_km
    return {
        "code": code,
        "label": label,
        "label_en": label_en,
        "label_km": label_km,
    }


class PredictRequest(BaseModel):
    symptoms: List[str] = Field(default_factory=list, description="List of symptom codes")


class PredictionScore(BaseModel):
    condition: str
    condition_label: str
    probability: float


class PredictResponse(BaseModel):
    locale: str
    predicted_condition: str
    predicted_condition_label: str
    confidence: float
    top_predictions: List[PredictionScore]
    active_symptom_count: int


app = FastAPI(title="Medical Symptom Condition Predictor", version="1.0.0")


def _get_allowed_origins() -> list[str]:
    configured = os.getenv("CORS_ALLOW_ORIGINS", "").strip()
    if configured:
        return [origin.strip().rstrip("/") for origin in configured.split(",") if origin.strip()]
    return [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://161.97.114.68:5173",
        "http://161.97.114.68:2343",
        "https://predictlogistic.leavchandara.site"
    ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_get_allowed_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model": metadata.get("best_model"),
        "features": len(features),
        "classes": len(classes),
        "supported_locales": list(SUPPORTED_LOCALES),
    }


@app.get("/symptoms")
def list_symptoms(lang: str | None = None) -> dict:
    locale = _normalize_locale(lang)
    return {
        "locale": locale,
        "supported_locales": list(SUPPORTED_LOCALES),
        "symptoms": [_symptom_payload(s, locale) for s in sorted(features)],
    }


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest, lang: str | None = None) -> PredictResponse:
    locale = _normalize_locale(lang)

    if not payload.symptoms:
        raise HTTPException(status_code=400, detail=_message("at_least_one_symptom", locale))

    unknown = [s for s in payload.symptoms if s not in features]
    if unknown:
        if locale == "km":
            unknown_preview = ", ".join(
                f"{_localize_symptom_label(symptom, locale)} [{symptom}]"
                for symptom in unknown[:10]
            )
        else:
            unknown_preview = ", ".join(unknown[:10])
        raise HTTPException(
            status_code=400,
            detail=f"{_message('unknown_symptoms_prefix', locale)}: {unknown_preview}",
        )

    row = {feature: 0 for feature in features}
    for symptom in payload.symptoms:
        row[symptom] = 1

    x = pd.DataFrame([row])

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(x)[0]
    else:
        scores = model.decision_function(x)
        if scores.ndim > 1:
            scores = scores[0]
        exp_scores = np.exp(scores - np.max(scores))
        probs = exp_scores / exp_scores.sum()

    top_idx = np.argsort(probs)[::-1][:3]
    top_predictions = [
        PredictionScore(
            condition=classes[i],
            condition_label=_localize_condition_label(classes[i], locale),
            probability=float(probs[i]),
        )
        for i in top_idx
    ]

    return PredictResponse(
        locale=locale,
        predicted_condition=top_predictions[0].condition,
        predicted_condition_label=top_predictions[0].condition_label,
        confidence=top_predictions[0].probability,
        top_predictions=top_predictions,
        active_symptom_count=len(payload.symptoms),
    )

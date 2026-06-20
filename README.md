# BankMind - Bank Marketing Subscription Predictor

Predicts whether a bank customer will subscribe to a term deposit. Three models trained, compared, and evaluated (Logistic Regression, Random Forest, XGBoost), with the final model served via FastAPI and a Groq LLM explanation layer on top.

**Track B (ML Engineer) + Track C (System Builder)** submission - VITBAIH Community Project 2026 Screening.

**Live demo:** https://bankmind-ishan.onrender.com/

---
**Note:** I also built a basic HTML form on top of the API, purely for easy manual testing - it chains the /predict and /explain calls together so you can see a full prediction plus explanation from one form submission instead of hitting each endpoint separately in Swagger.

---

## The ML Work (Track B)

This was the part I actually found most interesting to build, so I went a bit deeper than the minimum.

**The problem:** the dataset is heavily imbalanced - 88.3% of customers said no, only 11.7% said yes. A model can hit 90% accuracy by just guessing "no" every time, which is useless for a bank trying to find real leads. So accuracy alone isn't a real metric here - F1 and recall on the minority class are what actually matter.

**Three models, compared honestly:**

| Model | Accuracy | F1 (y=1) | Recall (y=1) | Precision (y=1) |
|---|---|---|---|---|
| Logistic Regression | 84.6% | 0.55 | 0.81 | 0.42 |
| Random Forest | 89.9% | 0.60 | 0.64 | 0.56 |
| **XGBoost (final)** | 87.8% | **0.61** | **0.81** | 0.49 |

Before fixing the class imbalance (via `class_weight='balanced'` / `scale_pos_weight`), Random Forest had 90.5% accuracy but only caught 39% of actual subscribers - that gap is the whole reason F1 matters more than accuracy here. After the fix, recall jumped to 64-81% across models at a small, deliberate cost to accuracy and precision.

**XGBoost was the final pick** - highest F1, and recall tied with Logistic Regression (0.81) but with much better precision (0.49 vs 0.42). For a bank calling leads, missing a real subscriber costs more than an unnecessary call, so recall was prioritized.

**Top feature driving predictions:** `poutcome_success` (0.1542) - whether the customer subscribed in a *previous* campaign. By far the strongest signal, well ahead of any demographic feature. Full reasoning, sample predictions, and EDA findings (e.g. students had the highest subscription rate at 28.68%) are in `EXPLANATION.md`.

---

## The API Layer (Track C)

The trained model is wrapped in a FastAPI service with three endpoints plus a bonus LLM explanation layer and a simple frontend form.

### Setup

```
git clone https://github.com/ishaann24/bankmind-ishan
cd bankmind-ishan
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file in the project root:
```
GROQ_API_KEY=your_groq_api_key_here
```

Train the model:
```
python train.py
```

Run the API:
```
python -m uvicorn main:app --reload
```

- Form UI: `http://localhost:8000/`
- Swagger docs: `http://localhost:8000/docs`

### Endpoints

**`GET /health`**
```
curl -X GET "http://localhost:8000/health"
```
```json
{"status": "ok", "model": "XGBoost"}
```

**`POST /predict`**
```
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "age": 64, "job": "retired", "marital": "married", "education": "secondary",
    "default": "no", "balance": 109, "housing": "no", "loan": "no",
    "contact": "cellular", "day": 5, "month": "may", "duration": 706,
    "campaign": 1, "pdays": -1, "previous": 0, "poutcome": "success"
  }'
```
```json
{
  "will_subscribe": true,
  "probability": 0.9908,
  "top_factors": ["previously subscribed to a campaign", "long call duration", "customer age profile"]
}
```

**`POST /explain`** (bonus - Groq LLM explanation)
```
curl -X POST "http://localhost:8000/explain" \
  -H "Content-Type: application/json" \
  -d '{"age": 64, "job": "retired", "balance": 109, "housing": "no", "loan": "no", "probability": 0.9908}'
```
```json
{"explanation": "This customer is likely to subscribe...", "probability": 0.9908}
```

**`GET /`** - Form UI, runs `/predict` and `/explain` together in one click.

**Note:** hosted on Render's free tier - first request after inactivity may take 30-50 seconds to wake up.

---

## Project Structure

```
bankmind-ishan/
├── data/bank-full.csv      # UCI Bank Marketing dataset
├── static/index.html       # Frontend form
├── explore.py               # EDA
├── preprocess.py            # Preprocessing + train/test split
├── train.py                 # Trains, evaluates, and saves all 3 models
├── main.py                  # FastAPI app
├── model.pkl / scaler.pkl / feature_names.json
├── requirements.txt
├── README.md
└── EXPLANATION.md
```

---

## Future Improvements

- Improve precision (currently 0.49) via threshold tuning or SMOTE oversampling
- Add stricter input validation on `/predict` (currently accepts any value for numeric/categorical fields)
- Make `/explain` async and add multiple uvicorn workers for concurrent load
- Add SHAP values for per-prediction interpretability

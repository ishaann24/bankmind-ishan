from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import joblib
import json
import pandas as pd
import numpy as np
from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

app = FastAPI(title="BankMind API", description="Predicts customer subscription likelihood")

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get('/')
def root():
    return FileResponse('static/index.html')

# Load model and feature names once at startup
model = joblib.load('model.pkl')
# ... rest of your existing code continues unchanged

# Load model and feature names once at startup
model = joblib.load('model.pkl')
with open('feature_names.json', 'r') as f:
    feature_names = json.load(f)

categorical_cols = ['job', 'marital', 'education', 'default',
                    'housing', 'loan', 'contact', 'month', 'poutcome']

# This defines exactly what JSON shape /predict expects
class Customer(BaseModel):
    age: int
    job: str
    marital: str
    education: str
    default: str
    balance: int
    housing: str
    loan: str
    contact: str
    day: int
    month: str
    duration: int
    campaign: int
    pdays: int
    previous: int
    poutcome: str

def preprocess_customer(customer: Customer) -> pd.DataFrame:
    # Convert incoming customer to dataframe
    data = pd.DataFrame([customer.dict()])
    
    # Apply same one-hot encoding as training
    data_encoded = pd.get_dummies(data, columns=categorical_cols, drop_first=True)
    
    # Reindex to match exact training columns - fills any missing columns with 0
    data_encoded = data_encoded.reindex(columns=feature_names, fill_value=0)
    
    return data_encoded.astype(int)

def get_top_factors(customer_df: pd.DataFrame) -> list:
    importances = model.feature_importances_
    customer_values = customer_df.values[0]
    
    # Weight importance by whether the feature is active for this customer
    weighted = importances * (customer_values != 0)
    top_indices = np.argsort(weighted)[::-1][:3]
    
    # Map raw feature names to readable descriptions
    readable = {
        'duration': 'long call duration',
        'balance': 'high account balance',
        'housing_yes': 'has housing loan',
        'loan_yes': 'has personal loan',
        'poutcome_success': 'previously subscribed to a campaign',
        'contact_unknown': 'unknown contact method',
        'default_yes': 'has credit default',
        'campaign': 'multiple campaign contacts',
        'age': 'customer age profile',
        'previous': 'prior campaign contacts',
    }
    
    factors = []
    for i in top_indices:
        name = feature_names[i]
        readable_name = readable.get(name, name.replace('_', ' '))
        factors.append(readable_name)
    
    return factors

@app.get('/health')
def health():
    return {"status": "ok", "model": "XGBoost"}

@app.post('/predict')
def predict(customer: Customer):
    customer_df = preprocess_customer(customer)
    prediction = model.predict(customer_df)[0]
    probability = model.predict_proba(customer_df)[0][1]
    top_factors = get_top_factors(customer_df)
    
    return {
        "will_subscribe": bool(prediction),
        "probability": round(float(probability), 4),
        "top_factors": top_factors
    }


class ExplainRequest(BaseModel):
    age: int
    job: str
    balance: int
    housing: str
    loan: str
    probability: float

@app.post('/explain')
def explain(request: ExplainRequest):
    prompt = f"""Customer profile:
- Age: {request.age}, Job: {request.job}, Balance: {request.balance}
- Existing loans: Housing={request.housing}, Personal={request.loan}
- Model prediction: {request.probability * 100:.1f}% chance of subscribing

In 2-3 sentences, explain why this customer would or would not likely subscribe to a term deposit, and how an RM should approach the conversation."""

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )
    
    explanation = response.choices[0].message.content
    
    return {
        "explanation": explanation,
        "probability": request.probability
    }
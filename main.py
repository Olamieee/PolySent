from fastapi import FastAPI, HTTPException, Depends, Request
from pydantic import BaseModel
from fastapi.security import APIKeyHeader
import re
import spacy
import string
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import json
import redis
import os
import logging
import sqlite3
from datetime import datetime, timedelta
import requests  # For exchange rate
from rave_python import Rave, RaveExceptions
import onnxruntime as ort  # Assuming this is imported elsewhere; add if needed
from dotenv import load_dotenv

# Initialize FastAPI
app = FastAPI(title="VibeSentry")
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Redis for caching (scalability)
redis_client = redis.Redis(host=os.getenv("REDIS_HOST", "localhost"), port=6379, db=0, decode_responses=True)

# Initialize Flutterwave
rave = Rave(os.getenv("FLW_PUBLIC_KEY"), 
            os.getenv("FLW_SECRET_KEY"),
            production=os.getenv("FLW_PRODUCTION", "False") == "True")

# Initialize SQLite for subscriptions (billing)
conn = sqlite3.connect("subscriptions.db", check_same_thread=False)
conn.execute("""
    CREATE TABLE IF NOT EXISTS subscriptions (
        api_key TEXT PRIMARY KEY,
        plan TEXT,
        requests_left INTEGER,
        expiry_date TEXT
    )
""")
conn.commit()

# Authentication (billing - verifies API key and decrements requests)
api_key_header = APIKeyHeader(name="X-API-Key")

async def verify_subscription(api_key: str = Depends(api_key_header)):
    cursor = conn.cursor()
    cursor.execute("SELECT plan, requests_left, expiry_date FROM subscriptions WHERE api_key = ?", (api_key,))
    result = cursor.fetchone()
    if not result or (result[1] != -1 and result[1] <= 0) or result[2] < datetime.now().isoformat():
        raise HTTPException(status_code=403, detail="Invalid or inactive subscription")
    # Decrement request count if not unlimited
    if result[1] != -1:
        cursor.execute("UPDATE subscriptions SET requests_left = requests_left - 1 WHERE api_key = ?", (api_key,))
        conn.commit()
    return api_key

# Load spaCy for English and Pidgin preprocessing
nlp = spacy.load("en_core_web_lg")
pidgin_stopwords = ["di", "abeg", "wetin", "sef", "abi", "dey", "na", "o", "sha", "joor"]
en_stopwords = nlp.Defaults.stop_words

# Load Keras models and tokenizers (fallback to ONNX if available)
max_seq_length = 128
MODEL_PATHS = {
    "pidgin": "./models/pidgin/pidgin.keras",
    "english": "./models/eng/english.keras",
    "yoruba": "./models/yoruba/yoruba.keras"
}
ONNX_PATHS = {
    "pidgin": "./models/pidgin/pidgin.onnx",
    "english": "./models/eng/english.onnx",
    "yoruba": "./models/yoruba/yoruba.onnx"
}

# Global variables for models (initialize here or in try block)
pidgin_session = None
english_session = None
yoruba_session = None
pidgin_model = None
english_model = None
yoruba_model = None
pidgin_tokenizer = None
english_tokenizer = None
yoruba_tokenizer = None

# Load models (Keras or ONNX)
try:
    # Try ONNX first for faster inference (scalability)
    pidgin_session = ort.InferenceSession(ONNX_PATHS["pidgin"]) if os.path.exists(ONNX_PATHS["pidgin"]) else None
    english_session = ort.InferenceSession(ONNX_PATHS["english"]) if os.path.exists(ONNX_PATHS["english"]) else None
    yoruba_session = ort.InferenceSession(ONNX_PATHS["yoruba"]) if os.path.exists(ONNX_PATHS["yoruba"]) else None
    
    # Fallback to Keras if ONNX not available
    if pidgin_session is None:
        pidgin_model = tf.keras.models.load_model(MODEL_PATHS["pidgin"])
    if english_session is None:
        english_model = tf.keras.models.load_model(MODEL_PATHS["english"])
    if yoruba_session is None:
        yoruba_model = tf.keras.models.load_model(MODEL_PATHS["yoruba"])
    
    # Load tokenizers
    with open("./models/pidgin/pidgin_tokenizer.json", "r", encoding="utf-8") as f:
        pidgin_tokenizer = tokenizer_from_json(f.read())
    with open("./models/eng/eng_tokenizer.json", "r", encoding="utf-8") as f:
        english_tokenizer = tokenizer_from_json(f.read())
    with open("./models/yoruba/tokenizer.json", "r", encoding="utf-8") as f:
        yoruba_tokenizer = tokenizer_from_json(f.read())
    logger.info("Models (ONNX or Keras) and tokenizers loaded successfully")
except Exception as e:
    logger.error(f"Failed to load models: {str(e)}")
    raise

# Label map for sentiment
label_map = {0: "negative", 1: "neutral", 2: "positive"}

# Pydantic model for input (language is required)
class TextInput(BaseModel):
    text: str
    language: str  # Now mandatory

# Preprocessing functions
def preprocess_text_eng(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return ""
    text = re.sub(r'@\w+|http\S+|www\S+|https\S+', '', text)
    doc = nlp(text)
    tokens = [token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct]
    return " ".join(tokens).strip() or "empty"

def preprocess_text_pidgin(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return ""
    text = re.sub(r'http\S+|@\w+|#\w+|\bRT\b|"{2,}', '', text, flags=re.IGNORECASE)
    text = re.sub(r'[.]{2,}', ' ', text)
    text = re.sub(r'[\!\?\:\.\,]+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    doc = nlp(text)
    tokens = [
        token.text.lower() for token in doc
        if token.text.lower() not in pidgin_stopwords
        and token.text.lower() not in en_stopwords
        and token.text not in string.punctuation
        and len(token.text.strip()) > 1
        and not token.text.isdigit()
    ]
    return " ".join(tokens).strip() or "empty"

def preprocess_text_yoruba(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return ""
    text = re.sub(r'http\S+|@\w+|#\w+', '', text)
    doc = nlp(text)
    tokens = [token.text.lower() for token in doc if not token.is_punct]
    return " ".join(tokens).strip() or "empty"

# Prediction function (supports ONNX or Keras for scalability)
def predict_keras(session, model, tokenizer, text: str) -> str:
    if not text or text == "empty":
        return "neutral"
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=max_seq_length, padding='post', truncating='post')
    if session:  # Use ONNX if available
        pred = session.run(None, {"input": padded.astype(np.float32)})[0]
    else:  # Fallback to Keras
        pred = model.predict(padded, verbose=0)
    return label_map[np.argmax(pred, axis=1)[0]]

# Function to get USD to NGN rate (using free, no-signup API)
def get_ngn_rate():
    # Check Redis cache first (1-hour expiry)
    cached_rate = redis_client.get("usd_ngn_rate")
    if cached_rate:
        return float(cached_rate)
    
    # Fetch from ExchangeRate.host (free, unlimited, no key)
    url = "https://api.exchangerate.host/latest?base=USD&symbols=NGN"
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            if data.get("success"):
                rate = data["rates"].get("NGN", 1600)  # Fallback to hardcoded if missing
                redis_client.setex("usd_ngn_rate", 3600, str(rate))  # Cache for 1 hour
                logger.info(f"Fetched NGN rate: {rate}")
                return float(rate)
    except Exception as e:
        logger.error(f"Exchange rate fetch failed: {str(e)}")
    
    # Ultimate fallback (update periodically based on current rates ~1530 as of 2025-09-22)
    fallback_rate = 1600
    redis_client.setex("usd_ngn_rate", 3600, str(fallback_rate))
    return fallback_rate

# Unified moderation endpoint with caching and authentication
@app.post("/moderate")
async def moderate_text(input: TextInput, api_key: str = Depends(verify_subscription)):
    try:
        # Check cache (scalability)
        cache_key = f"{input.language}:{input.text}"
        cached_result = redis_client.get(cache_key)
        if cached_result:
            logger.info("Returning cached result")
            return json.loads(cached_result)

        # Route to model based on selected language
        if input.language.lower() == "yoruba":
            cleaned_text = preprocess_text_yoruba(input.text)
            result = {"sentiment": predict_keras(yoruba_session, yoruba_model, yoruba_tokenizer, cleaned_text)}
        elif input.language.lower() == "english":
            cleaned_text = preprocess_text_eng(input.text)
            result = {"sentiment": predict_keras(english_session, english_model, english_tokenizer, cleaned_text)}
        elif input.language.lower() == "pidgin":
            cleaned_text = preprocess_text_pidgin(input.text)
            result = {"sentiment": predict_keras(pidgin_session, pidgin_model, pidgin_tokenizer, cleaned_text)}
        else:
            raise HTTPException(status_code=400, detail="Unsupported language: choose 'English', 'Pidgin', or 'Yoruba'")

        # Cache result for 1 hour (scalability)
        redis_client.setex(cache_key, 3600, json.dumps(result))
        return result
    except Exception as e:
        logger.error(f"Error processing text: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "Content Moderation API - Use /moderate to analyze text"}

# Endpoint to create a free trial (for testing)
@app.post("/create_trial")
async def create_trial():
    api_key = "trial_" + str(datetime.now().timestamp())  # Generate unique key
    expiry = (datetime.now() + timedelta(days=14)).isoformat()
    cursor = conn.cursor()
    cursor.execute("INSERT INTO subscriptions (api_key, plan, requests_left, expiry_date) VALUES (?, ?, ?, ?)",
                   (api_key, "trial", 1000, expiry))
    conn.commit()
    return {"api_key": api_key, "expiry": expiry}

# Subscription input model
class SubscribeInput(BaseModel):
    plan: str
    email: str

# Endpoint to initiate subscription
@app.post("/subscribe")
async def subscribe(input: SubscribeInput):
    usd_prices = {"basic": 10, "pro": 50}
    if input.plan not in usd_prices:
        raise HTTPException(status_code=400, detail="Invalid plan")
    
    ngn_rate = get_ngn_rate()
    amount = int(usd_prices[input.plan] * ngn_rate)  # Convert to NGN (int for Flutterwave)
    
    payload = {
        "amount": amount,
        "email": input.email,
        "currency": "NGN",
        "redirect_url": os.getenv("CALLBACK_URL", "http://localhost:5000/callback"),
        "meta": {"plan": input.plan},
        "txref": "vibesentry_" + str(datetime.now().timestamp())
    }
    
    try:
        res = rave.Account.charge(payload)  # Using account charge for simplicity; adjust for card if needed
        if res.get("authUrl"):
            return {"payment_url": res["authUrl"]}
        elif res.get("validationRequired"):
            # Handle further validation if needed
            raise RaveExceptions.TransactionChargeError("Validation required")
    except RaveExceptions.TransactionChargeError as e:
        raise HTTPException(status_code=500, detail=str(e.err))
    raise HTTPException(status_code=500, detail="Payment initiation failed")

# Webhook for Flutterwave
@app.post("/webhook")
async def webhook(request: Request):
    secret_hash = os.getenv("FLW_WEBHOOK_HASH")
    signature = request.headers.get("verif-hash")
    if signature != secret_hash:
        raise HTTPException(status_code=401, detail="Invalid webhook signature")

    data = await request.json()
    if data['event'] == "charge.completed" and data['data']['status'] == "successful":
        transaction = data['data']
        plan = transaction['meta']['plan']
        email = transaction['customer']['email']
        api_key = "sub_" + str(datetime.now().timestamp())
        requests_left = 10000 if plan == "basic" else -1  # -1 for unlimited
        expiry = (datetime.now() + timedelta(days=30)).isoformat()
        cursor = conn.cursor()
        cursor.execute("INSERT INTO subscriptions (api_key, plan, requests_left, expiry_date) VALUES (?, ?, ?, ?)",
                       (api_key, plan, requests_left, expiry))
        conn.commit()
        # In production, send email with api_key to user
        logger.info(f"Subscription created for {email}: API Key={api_key}")
    return {"status": "success"}
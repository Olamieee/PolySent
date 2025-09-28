from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
import re
import string
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import json
import os
import logging
from logging.handlers import RotatingFileHandler
import sqlite3
from datetime import datetime, timedelta
import requests
from rave_python import Rave, RaveExceptions
import onnxruntime as ort
from dotenv import load_dotenv, find_dotenv

# Suppress TensorFlow INFO and WARNING logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Load environment variables
env_file = find_dotenv()
if env_file:
    load_dotenv(env_file)
    logging.info(f"Loaded .env file from {env_file}")
else:
    logging.warning("No .env file found; relying on system environment variables")

# Flask app initialization
app = Flask(__name__)

# Configuration classes
class Config:
    SECRET_KEY = os.getenv("FLASK_SECRET_KEY")
    if not SECRET_KEY:
        logging.error("FLASK_SECRET_KEY not found in environment")
        SECRET_KEY = "your_secret_key_dev"
    FLW_PUBLIC_KEY = os.getenv("FLW_PUBLIC_KEY") if os.getenv("FLW_PRODUCTION", "False") == "True" else os.getenv("FLW_TEST_PUBLIC")
    if not FLW_PUBLIC_KEY:
        logging.error("FLW_PUBLIC_KEY or FLW_TEST_PUBLIC not found in environment")
        FLW_PUBLIC_KEY = "your_flw_public_key_dev"
    FLW_SECRET_KEY = os.getenv("FLW_SECRET_KEY") if os.getenv("FLW_PRODUCTION", "False") == "True" else os.getenv("FLW_TEST_SECRET")
    if not FLW_SECRET_KEY:
        logging.error("FLW_SECRET_KEY or FLW_TEST_SECRET not found in environment")
        FLW_SECRET_KEY = "your_flw_secret_key_dev"
    FLW_PRODUCTION = os.getenv("FLW_PRODUCTION", "False") == "True"
    FLW_WEBHOOK_HASH = os.getenv("FLW_WEBHOOK_HASH")
    if not FLW_WEBHOOK_HASH:
        logging.error("FLW_WEBHOOK_HASH not found in environment")
        FLW_WEBHOOK_HASH = "your_webhook_hash_dev"
    CALLBACK_URL = os.getenv("CALLBACK_URL", "http://localhost:5000/callback")

class DevelopmentConfig(Config):
    DEBUG = True
    DATABASE = os.getenv("DATABASE", "subscriptions.db")
    LOG_FILE = "vibesentry_dev.log"

class ProductionConfig(Config):
    DEBUG = False
    DATABASE = os.getenv("DATABASE", "/path/to/production/subscriptions.db")
    LOG_FILE = "/var/log/vibesentry.log"

# Select config based on environment
env = os.getenv("FLASK_ENV", "development")
app.config.from_object(DevelopmentConfig if env == "development" else ProductionConfig)
logging.info(f"Running in {env} mode")

# Logging setup
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
if env == "development":
    handler = logging.StreamHandler()
else:
    handler = RotatingFileHandler(app.config['LOG_FILE'], maxBytes=1000000, backupCount=5)
handler.setFormatter(formatter)
logger.addHandler(handler)

# Initialize Flutterwave
try:
    rave = Rave(
        app.config['FLW_PUBLIC_KEY'],
        app.config['FLW_SECRET_KEY'],
        production=app.config['FLW_PRODUCTION'],
        usingEnv=False
    )
    logger.info("Flutterwave initialized")
except Exception as e:
    logger.error(f"Failed to initialize Flutterwave: {str(e)}")
    raise

# Initialize SQLite
def get_db_connection():
    try:
        conn = sqlite3.connect(app.config['DATABASE'], check_same_thread=False)
        # Updated schema to include user information
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                first_name TEXT,
                last_name TEXT,
                email TEXT UNIQUE,
                country TEXT,
                created_at TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS subscriptions (
                api_key TEXT PRIMARY KEY,
                user_id INTEGER,
                plan TEXT,
                requests_left INTEGER,
                expiry_date TEXT,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS moderation_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                api_key TEXT,
                text TEXT,
                language TEXT,
                sentiment TEXT,
                timestamp TEXT
            )
        """)
        conn.commit()
        logger.info(f"Connected to database at {app.config['DATABASE']}")
        return conn
    except Exception as e:
        logger.error(f"Failed to connect to database: {str(e)}")
        raise

conn = get_db_connection()

# Stop words and preprocessing setup
pidgin_stopwords = ["di", "abeg", "wetin", "sef", "abi", "dey", "na", "o", "sha", "joor"]
en_stopwords = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "he",
    "in", "is", "it", "its", "of", "on", "that", "the", "to", "was", "were", "will",
    "with", "you", "your", "this", "but", "if", "or", "because", "until", "while",
    "about", "against", "between", "into", "through", "during", "before", "after",
    "above", "below", "up", "down", "out", "off", "over", "under", "again", "further",
    "then", "once", "here", "there", "when", "where", "why", "how", "all", "any",
    "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor",
    "not", "only", "own", "same", "so", "than", "too", "very"
}

# Model paths and loading
max_seq_length = 50
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

# Global variables for models
pidgin_session = None
english_session = None
yoruba_session = None
pidgin_model = None
english_model = None
yoruba_model = None
pidgin_tokenizer = None
english_tokenizer = None
yoruba_tokenizer = None

# Load models
try:
    pidgin_session = ort.InferenceSession(ONNX_PATHS["pidgin"]) if os.path.exists(ONNX_PATHS["pidgin"]) else None
    english_session = ort.InferenceSession(ONNX_PATHS["english"]) if os.path.exists(ONNX_PATHS["english"]) else None
    yoruba_session = ort.InferenceSession(ONNX_PATHS["yoruba"]) if os.path.exists(ONNX_PATHS["yoruba"]) else None
    
    if pidgin_session is None:
        pidgin_model = tf.keras.models.load_model(MODEL_PATHS["pidgin"])
    if english_session is None:
        english_model = tf.keras.models.load_model(MODEL_PATHS["english"])
    if yoruba_session is None:
        yoruba_model = tf.keras.models.load_model(MODEL_PATHS["yoruba"])
    
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

# Preprocessing functions
def preprocess_text_eng(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return ""
    text = re.sub(r'@\w+|http\S+|www\S+|https\S+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation)).lower()
    tokens = re.split(r'\s+', text.strip())
    tokens = [token for token in tokens if token not in en_stopwords]
    return " ".join(tokens).strip() or "empty"

def preprocess_text_pidgin(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return ""
    text = re.sub(r'http\S+|@\w+|#\w+|\bRT\b|"{2,}', '', text, flags=re.IGNORECASE)
    text = re.sub(r'[.]{2,}', ' ', text)
    text = re.sub(r'[\!\?\:\.\,]+', ' ', text)
    text = text.lower().strip()
    tokens = re.split(r'\s+', text)
    tokens = [token for token in tokens if token not in pidgin_stopwords and token not in en_stopwords]
    return " ".join(tokens).strip() or "empty"

def preprocess_text_yoruba(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return ""
    text = re.sub(r'http\S+|@\w+|#\w+|\bRT\b|"{2,}', '', text, flags=re.IGNORECASE)
    text = text.lower().translate(str.maketrans('', '', string.punctuation))
    tokens = re.split(r'\s+', text.strip())
    return " ".join(tokens).strip() or "empty"

# Prediction function
def predict_keras(session, model, tokenizer, text: str):
    sequences = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequences, maxlen=max_seq_length, padding='post', truncating='post')
    if session is not None:
        input_name = session.get_inputs()[0].name
        pred = session.run(None, {input_name: padded.astype(np.float32)})[0]
    else:
        pred = model.predict(padded)[0]
    return label_map[np.argmax(pred)]

# Verify subscription
def verify_subscription(api_key: str):
    cursor = conn.cursor()
    cursor.execute("SELECT plan, requests_left, expiry_date FROM subscriptions WHERE api_key = ?", (api_key,))
    result = cursor.fetchone()
    if not result or (result[1] != -1 and result[1] <= 0) or result[2] < datetime.now().isoformat():
        return False
    if result[1] != -1:
        cursor.execute("UPDATE subscriptions SET requests_left = requests_left - 1 WHERE api_key = ?", (api_key,))
        conn.commit()
    return True

# Get user info by API key
def get_user_by_api_key(api_key: str):
    cursor = conn.cursor()
    cursor.execute("""
        SELECT u.first_name, u.last_name, u.email, s.plan, s.requests_left, s.expiry_date 
        FROM users u 
        JOIN subscriptions s ON u.id = s.user_id 
        WHERE s.api_key = ?
    """, (api_key,))
    return cursor.fetchone()

# Log moderation result to DB
def log_moderation(api_key: str, text: str, language: str, sentiment: str):
    try:
        cursor = conn.cursor()
        timestamp = datetime.now().isoformat()
        cursor.execute("INSERT INTO moderation_logs (api_key, text, language, sentiment, timestamp) VALUES (?, ?, ?, ?, ?)",
                      (api_key, text, language, sentiment, timestamp))
        conn.commit()
        logger.info(f"Logged moderation for API key {api_key}")
    except Exception as e:
        logger.error(f"Failed to log moderation: {str(e)}")

# Get NGN exchange rate
def get_ngn_rate():
    url = "https://api.exchangerate-api.com/v4/latest/USD"
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            if 'rates' in data:
                rate = data["rates"].get("NGN", 1530)
                logger.info(f"Fetched NGN rate: {rate}")
                return float(rate)
    except Exception as e:
        logger.error(f"Exchange rate fetch failed: {str(e)}")
    fallback_rate = 1530
    logger.info(f"Using fallback NGN rate: {fallback_rate}")
    return fallback_rate

# Error handlers
@app.errorhandler(404)
def page_not_found(e):
    flash("Page not found. Please check the URL and try again.", "error")
    return render_template("error.html", session=session), 404

@app.errorhandler(500)
def internal_server_error(e):
    flash("An unexpected error occurred. Please try again later.", "error")
    return render_template("error.html", session=session), 500

# Routes
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", session=session)

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        first_name = request.form.get("first_name")
        last_name = request.form.get("last_name")
        email = request.form.get("email")
        country = request.form.get("country")
        
        if not all([first_name, last_name, email, country]):
            flash("All fields are required", "error")
            return render_template("signup.html", session=session)
        
        try:
            cursor = conn.cursor()
            # Check if email already exists
            cursor.execute("SELECT id FROM users WHERE email = ?", (email,))
            if cursor.fetchone():
                flash("Email already registered. Please use a different email or login.", "error")
                return render_template("signup.html", session=session)
            
            # Insert new user
            cursor.execute("""
                INSERT INTO users (first_name, last_name, email, country, created_at) 
                VALUES (?, ?, ?, ?, ?)
            """, (first_name, last_name, email, country, datetime.now().isoformat()))
            conn.commit()
            
            # Store user info in session for pricing page
            session["signup_user"] = {
                "first_name": first_name,
                "last_name": last_name,
                "email": email,
                "country": country
            }
            
            flash("Account created successfully! Please choose a plan to continue.", "success")
            return redirect(url_for("pricing"))
            
        except Exception as e:
            logger.error(f"Signup error: {str(e)}")
            flash("An error occurred during signup. Please try again.", "error")
            return render_template("signup.html", session=session)
    
    return render_template("signup.html", session=session)

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email")
        api_key = request.form.get("api_key")
        
        if not all([email, api_key]):
            flash("Both email and API key are required", "error")
            return render_template("login.html", session=session)
        
        cursor = conn.cursor()
        cursor.execute("""
            SELECT u.id, u.first_name, u.last_name, u.email, s.plan, s.requests_left, s.expiry_date 
            FROM users u 
            JOIN subscriptions s ON u.id = s.user_id 
            WHERE u.email = ? AND s.api_key = ?
        """, (email, api_key))
        result = cursor.fetchone()
        
        if result and result[6] > datetime.now().isoformat():
            session["api_key"] = api_key
            session["user_info"] = {
                "first_name": result[1],
                "last_name": result[2],
                "email": result[3],
                "plan": result[4]
            }
            return redirect(url_for("dashboard"))
        
        flash("Invalid email or API key, or subscription expired", "error")
        return render_template("login.html", session=session)
    
    return render_template("login.html", session=session)

@app.route("/dashboard", methods=["GET", "POST"])
def dashboard():
    if "api_key" not in session:
        return redirect(url_for("login"))
    
    result = None
    if request.method == "POST":
        text = request.form.get("text")
        language = request.form.get("language")
        if not language:
            result = {"error": "Please select a language"}
        else:
            api_key = session["api_key"]
            if not verify_subscription(api_key):
                result = {"error": "Invalid or inactive subscription"}
            else:
                try:
                    if language.lower() == "yoruba":
                        cleaned_text = preprocess_text_yoruba(text)
                        sentiment = predict_keras(yoruba_session, yoruba_model, yoruba_tokenizer, cleaned_text)
                    elif language.lower() == "english":
                        cleaned_text = preprocess_text_eng(text)
                        sentiment = predict_keras(english_session, english_model, english_tokenizer, cleaned_text)
                    elif language.lower() == "pidgin":
                        cleaned_text = preprocess_text_pidgin(text)
                        sentiment = predict_keras(pidgin_session, pidgin_model, pidgin_tokenizer, cleaned_text)
                    else:
                        result = {"error": "Unsupported language: choose 'English', 'Pidgin', or 'Yoruba'"}
                        return render_template("dashboard.html", result=result, session=session)
                    result = {"sentiment": sentiment}
                    log_moderation(api_key, text, language, sentiment)
                except Exception as e:
                    logger.error(f"Error processing text: {str(e)}")
                    result = {"error": str(e)}
    
    return render_template("dashboard.html", result=result, session=session)

@app.route("/analytics")
def analytics():
    if "api_key" not in session:
        return redirect(url_for("login"))
    
    cursor = conn.cursor()
    cursor.execute("SELECT sentiment FROM moderation_logs WHERE api_key = ? ORDER BY timestamp DESC LIMIT 100", (session["api_key"],))
    logs = cursor.fetchall()
    
    if logs:
        negative_count = sum(1 for log in logs if log[0] == "negative")
        neutral_count = sum(1 for log in logs if log[0] == "neutral")
        positive_count = sum(1 for log in logs if log[0] == "positive")
        total = len(logs)
        analytics = {
            "negative": round((negative_count / total) * 100, 2) if total > 0 else 0,
            "neutral": round((neutral_count / total) * 100, 2) if total > 0 else 0,
            "positive": round((positive_count / total) * 100, 2) if total > 0 else 0
        }
        logger.info(f"Analytics fetched for {session['api_key']}: {analytics}")
    else:
        analytics = {"negative": 0, "neutral": 0, "positive": 0}
        flash("No entries yet", "info")
        logger.info(f"No logs found for {session['api_key']}")
    
    return render_template("analytics.html", analytics=analytics, session=session)

@app.route("/logout")
def logout():
    session.clear()
    flash("You have been logged out successfully.", "success")
    return redirect(url_for("index"))

@app.route("/pricing")
def pricing():
    return render_template("pricing.html", session=session)

@app.route("/contact", methods=["GET", "POST"])
def contact():
    if request.method == "POST":
        name = request.form.get("name")
        email = request.form.get("email")
        message = request.form.get("message")
        logger.info(f"Contact form submitted: Name={name}, Email={email}, Message={message}")
        flash("Message sent successfully!", "success")
        return redirect(url_for("contact"))
    return render_template("contact.html", session=session)

@app.route("/create_trial")
def create_trial():
    # Check if user signed up or get info from session
    user_info = session.get("signup_user")
    
    if not user_info:
        # If no signup info, create anonymous trial
        api_key = "trial_" + str(datetime.now().timestamp())
        expiry = (datetime.now() + timedelta(days=14)).isoformat()
        cursor = conn.cursor()
        cursor.execute("INSERT INTO subscriptions (api_key, plan, requests_left, expiry_date) VALUES (?, ?, ?, ?)",
                       (api_key, "trial", 1000, expiry))
        conn.commit()
        trial = {"api_key": api_key, "expiry": expiry}
        return render_template("trial.html", trial=trial, session=session)
    
    # Create trial for signed up user
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM users WHERE email = ?", (user_info["email"],))
    user_result = cursor.fetchone()
    
    if user_result:
        user_id = user_result[0]
        api_key = "trial_" + str(datetime.now().timestamp())
        expiry = (datetime.now() + timedelta(days=14)).isoformat()
        
        cursor.execute("INSERT INTO subscriptions (api_key, user_id, plan, requests_left, expiry_date) VALUES (?, ?, ?, ?, ?)",
                       (api_key, user_id, "trial", 1000, expiry))
        conn.commit()
        
        # Clear signup session
        session.pop("signup_user", None)
        
        trial = {"api_key": api_key, "expiry": expiry, "email": user_info["email"]}
        return render_template("trial.html", trial=trial, session=session)
    
    flash("User not found. Please sign up again.", "error")
    return redirect(url_for("signup"))

@app.route("/subscribe", methods=["POST"])
def subscribe():
    plan = request.form.get("plan")
    email = request.form.get("email")
    usd_prices = {"basic": 10, "pro": 50}
    
    if plan not in usd_prices:
        flash("Invalid plan", "error")
        return redirect(url_for("pricing"))
    
    # Check if user is from signup flow
    user_info = session.get("signup_user")
    if user_info:
        email = user_info["email"]  # Use email from signup
    
    # Get user info from database
    cursor = conn.cursor()
    cursor.execute("SELECT id, first_name, last_name FROM users WHERE email = ?", (email,))
    user_result = cursor.fetchone()
    
    if not user_result:
        flash("Please sign up first before subscribing.", "error")
        return redirect(url_for("signup"))
    
    user_id, first_name, last_name = user_result
    
    ngn_rate = get_ngn_rate()
    amount = int(usd_prices[plan] * ngn_rate)
    
    payload = {
        "amount": amount,
        "email": email,
        "currency": "NGN",
        "redirect_url": app.config['CALLBACK_URL'],
        "customer": {
            "email": email,
            "firstname": first_name,
            "lastname": last_name
        },
        "meta": {
            "plan": plan,
            "user_id": user_id,
            "first_name": first_name,
            "last_name": last_name
        },
        "txref": "vibesentry_" + str(datetime.now().timestamp())
    }
    
    try:
        res = rave.Account.charge(payload)
        if res.get("authUrl"):
            return redirect(res["authUrl"])
        elif res.get("validationRequired"):
            flash("Validation required", "error")
            return redirect(url_for("pricing"))
    except RaveExceptions.TransactionChargeError as e:
        flash(str(e.err), "error")
        return redirect(url_for("pricing"))
    flash("Payment initiation failed", "error")
    return redirect(url_for("pricing"))

@app.route("/callback")
def callback():
    flash("Payment processed. Check your email for confirmation.", "success")
    return redirect(url_for("login"))

@app.route("/webhook", methods=["POST"])
def webhook():
    secret_hash = app.config['FLW_WEBHOOK_HASH']
    signature = request.headers.get("verif-hash")
    if signature != secret_hash:
        return jsonify({"status": "error", "detail": "Invalid webhook signature"}), 401

    data = request.json
    if data['event'] == "charge.completed" and data['data']['status'] == "successful":
        transaction = data['data']
        plan = transaction['meta']['plan']
        user_id = transaction['meta']['user_id']
        email = transaction['customer']['email']
        api_key = "sub_" + str(datetime.now().timestamp())
        requests_left = 10000 if plan == "basic" else -1
        expiry = (datetime.now() + timedelta(days=30)).isoformat()
        
        cursor = conn.cursor()
        cursor.execute("INSERT INTO subscriptions (api_key, user_id, plan, requests_left, expiry_date) VALUES (?, ?, ?, ?, ?)",
                       (api_key, user_id, plan, requests_left, expiry))
        conn.commit()
        logger.info(f"Subscription created for {email}: API Key={api_key}")
        
        # Clear signup session if exists
        session.pop("signup_user", None)
    return jsonify({"status": "success"})

@app.route("/moderate", methods=["POST"])
def moderate_text():
    data = request.json
    if not data or 'text' not in data or 'language' not in data:
        return jsonify({"error": "Missing text or language"}), 400
    
    text = data['text']
    language = data['language']
    api_key = request.headers.get("X-API-Key")
    if not api_key or not verify_subscription(api_key):
        return jsonify({"error": "Invalid or inactive subscription"}), 403
    
    try:
        if language.lower() == "yoruba":
            cleaned_text = preprocess_text_yoruba(text)
            sentiment = predict_keras(yoruba_session, yoruba_model, yoruba_tokenizer, cleaned_text)
        elif language.lower() == "english":
            cleaned_text = preprocess_text_eng(text)
            sentiment = predict_keras(english_session, english_model, english_tokenizer, cleaned_text)
        elif language.lower() == "pidgin":
            cleaned_text = preprocess_text_pidgin(text)
            sentiment = predict_keras(pidgin_session, pidgin_model, pidgin_tokenizer, cleaned_text)
        else:
            return jsonify({"error": "Unsupported language: choose 'English', 'Pidgin', or 'Yoruba'"}), 400
        
        result = {"sentiment": sentiment}
        log_moderation(api_key, text, language, sentiment)
        return result
    except Exception as e:
        logger.error(f"Error processing text: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    if env == "development":
        app.run(host="0.0.0.0", port=5000, debug=True)
    else:
        logger.info("Production mode: Use a WSGI server like Gunicorn")
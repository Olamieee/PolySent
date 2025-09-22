from flask import Flask, render_template, request, redirect, url_for, session, flash
import requests
import sqlite3
from datetime import datetime
import os
import logging

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "your_secret_key")
API_URL = os.getenv("API_URL", "http://localhost:8000/moderate")  # Point to FastAPI

# Initialize SQLite (shared with FastAPI for subscriptions)
conn = sqlite3.connect("subscriptions.db", check_same_thread=False)

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        api_key = request.form.get("api_key")
        cursor = conn.cursor()
        cursor.execute("SELECT plan, requests_left, expiry_date FROM subscriptions WHERE api_key = ?", (api_key,))
        result = cursor.fetchone()
        if result and result[2] > datetime.now().isoformat():
            session["api_key"] = api_key
            return redirect(url_for("dashboard"))
        flash("Invalid or expired API key", "error")
        return render_template("login.html")
    return render_template("login.html")

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
            headers = {"X-API-Key": session["api_key"]}
            response = requests.post(API_URL, json={"text": text, "language": language}, headers=headers)
            if response.status_code == 200:
                result = response.json()
            else:
                result = {"error": response.json().get("detail", "API error")}
    
    return render_template("dashboard.html", result=result)

@app.route("/analytics")
def analytics():
    if "api_key" not in session:
        return redirect(url_for("login"))
    # Mock analytics - replace with real data from DB in production
    analytics = {"negative": 10, "neutral": 60, "positive": 30}
    return render_template("analytics.html", analytics=analytics)

@app.route("/logout")
def logout():
    session.pop("api_key", None)
    return redirect(url_for("login"))

@app.route("/pricing")
def pricing():
    return render_template("pricing.html")

@app.route("/contact", methods=["GET", "POST"])
def contact():
    if request.method == "POST":
        name = request.form.get("name")
        email = request.form.get("email")
        message = request.form.get("message")
        # In production, integrate email service like SendGrid
        logger.info(f"Contact form submitted: Name={name}, Email={email}, Message={message}")
        flash("Message sent successfully!", "success")
        return redirect(url_for("contact"))
    return render_template("contact.html")

@app.route("/create_trial")
def create_trial():
    response = requests.post(API_URL.replace("/moderate", "/create_trial"))
    if response.status_code == 200:
        trial = response.json()
        return render_template("trial.html", trial=trial)
    flash("Error creating trial", "error")
    return redirect(url_for("index"))

@app.route("/subscribe", methods=["POST"])
def subscribe():
    plan = request.form.get("plan")
    email = request.form.get("email")
    response = requests.post(API_URL.replace("/moderate", "/subscribe"), json={"plan": plan, "email": email})
    if response.status_code == 200:
        return redirect(response.json()["payment_url"])
    flash("Error initiating subscription", "error")
    return redirect(url_for("pricing"))

@app.route("/callback")
def callback():
    # Handle payment callback if needed, but webhook in FastAPI handles fulfillment
    flash("Payment processed. Check your email for API key.", "success")
    return redirect(url_for("login"))

if __name__ == "__main__":
    # For production, use gunicorn: gunicorn -w 4 app:app
    app.run(host="0.0.0.0", port=5000, debug=True)
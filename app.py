import streamlit as st 
import time
import joblib
import pandas as pd
import numpy as np
import os
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from firebase_config import db
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from io import BytesIO


CHATBOT_SUGGESTED_QUESTIONS = [
    "What is DiagnoBridge?",
    "How do I use this app?",
    "How do I navigate the tabs?",
    "Which prediction tool should I use?",
    "How do I check my prediction history?",
    "How can I download my health report?",
    "Explain symptom-based prediction",
    "What can the chatbot help me with?"
]

# =====================================================
def ollama_medical_reply(user_text: str, chat_context: str = ""):
    try:
        payload = {
            "model": "llama3",
            "prompt": f"""
You are a medical education assistant inside a health prediction app called DiagnoBridge.

Rules:
- Educational information only
- No diagnosis or treatment
- Simple language
- Short explanation (max 100 words)
- DO NOT ask follow-up questions
- DO NOT include phrases like "You can also ask"
- End with a medical disclaimer

Conversation context:
{chat_context}

User question:
{user_text}
""",
            "stream": False
        }

        r = requests.post(
            "http://localhost:1143/api/generate",
            json=payload,
            timeout=120
        )

        return r.json()["response"].strip()

    except Exception:
        return (
            "⚠️ Local AI is not responding.\n\n"
            "Please ensure Ollama is running (`ollama serve`) and try again."
        )



# =====================================================
# CHATBOT KNOWLEDGE BASE (RULE-BASED)
# =====================================================

CHATBOT_KNOWLEDGE_BASE = {

    "app_overview": {
        "app_only": True,
        "keywords": [
            "what is this app",
            "what does this app do",
            "diagnobridge"
        ],
        "answer": (
            "🩺 **DiagnoBridge** is a health prediction and monitoring app. "
            "It estimates health risks using machine learning models for "
            "Diabetes, Heart Disease, Parkinson’s, and symptom-based conditions.\n\n"
            "💡 *Tip: Use the top buttons to switch between Prediction Tool, History, and Chatbot anytime.*"
        )
    },

    "how_to_use": {
        "app_only": True,
        "keywords": [
            "how to use",
            "use this app",
            "how do i use"
        ],
        "answer": (
            "📌 **How to use DiagnoBridge:**\n\n"
            "1️⃣ Choose a prediction tool from the sidebar\n"
            "2️⃣ Enter the required health details or symptoms\n"
            "3️⃣ Click **Predict**\n"
            "4️⃣ View risk results and general medical tips\n"
            "5️⃣ Open **Prediction History** to review past predictions\n\n"
            "💡 *Tip: Use the top buttons to switch between Prediction Tool, History, and Chatbot anytime.*"
        )
    },

    "use_disease_prediction": {
        "app_only": True,
        "priority": 1,
        "keywords": [
            "how do i use disease prediction",
            "use disease prediction",
            "how does disease prediction work"
        ],
        "answer": (
            "🧪 **Using Disease Prediction**\n\n"
            "• Select a disease (Diabetes, Heart, Parkinson’s, or Symptom-based)\n"
            "• Fill in the required medical values or symptoms\n"
            "• Click **Predict** to get a risk estimate\n"
            "• Review the result and medical tips shown below\n\n"
            "⚠️ This is an AI-based prediction, not a medical diagnosis.\n\n"
            "💡 *Tip: Use the top buttons to switch between Prediction Tool, History, and Chatbot anytime.*"
        )
    },

    "which_prediction": {
        "app_only": True,
        "priority": 2,
        "keywords": [
            "which prediction should i use",
            "what prediction should i use",
            "which disease prediction"
        ],
        "answer": (
            "🧭 **Choosing the Right Prediction Tool**\n\n"
            "• 🩸 **Diabetes Prediction** → If you know glucose, BMI, BP, cholesterol\n"
            "• ❤️ **Heart Disease Prediction** → If you have BP, cholesterol, heart rate data\n"
            "• 🧠 **Parkinson’s Prediction** → If voice feature values are available\n"
            "• 🧾 **Symptom-based Prediction** → If you only know symptoms\n\n"
            "If unsure, start with **Symptom-based Prediction**.\n\n"
            "💡 *Tip: Use the top buttons to switch between Prediction Tool, History, and Chatbot anytime.*"
        )
    },

    "why_use_app": {
        "app_only": True,
        "priority": 1,
        "keywords": [
            "why should i use this app",
            "why use this app",
            "why use diagnobridge",
            "benefit of this app"
        ],
        "answer": (
            "🌟 **Why Use DiagnoBridge?**\n\n"
            "- Get **quick health risk insights** using ML models\n"
            "- Track **prediction history** over time\n"
            "- Use **symptom-based prediction** when tests are unavailable\n"
            "- Download **PDF health reports**\n"
            "- Simple, beginner-friendly interface\n\n"
            "⚠️ This app provides **risk estimation**, not medical diagnosis."
        )
    },

    "enter_symptoms": {
        "app_only": True,
        "keywords": [
            "how do i enter symptoms",
            "enter symptoms",
            "symptom input",
            "add symptoms"
        ],
        "answer": (
            "✍️ **How to Enter Symptoms**\n\n"
            "You can enter symptoms in two ways:\n\n"
            "1️⃣ **Checklist Mode** – Select symptoms from a list\n"
            "2️⃣ **Free-text Mode** – Describe symptoms in your own words\n"
            "   Example: *Fever, headache, body pain for 2 days*\n\n"
            "💡 *Tip: Use the top buttons to switch between Prediction Tool, History, and Chatbot anytime.*"
        )
    },

    "prediction_history": {
        "app_only": True,
        "priority": 3,
        "keywords": [
            "prediction history",
            "my history",
            "past predictions",
            "saved results",
            "saved predictions",
            "where are my results",
            "show my prediction history"
        ],
        "answer": (
            "📊 **Prediction History** stores all your past predictions.\n\n"
            "• View previous results\n"
            "• Track risk trends over time\n"
            "• Download PDF health reports\n\n"
            "🧭 Access it from the **Prediction History** tab at the top.\n\n"
            "💡 *Tip: Use the top buttons to switch between Prediction Tool, History, and Chatbot anytime.*"
        )
    },

    "download_report": {
        "app_only": True,
        "keywords": [
            "download report",
            "download pdf",
            "health report",
            "export report"
        ],
        "answer": (
            "📥 **Downloading Your Health Report**\n\n"
            "1️⃣ Open **Prediction History**\n"
            "2️⃣ Apply filters if needed\n"
            "3️⃣ Scroll to **Download PDF Reports**\n"
            "4️⃣ Click **Download** for the required disease\n\n"
            "💡 *Tip: Use the top buttons to switch between Prediction Tool, History, and Chatbot anytime.*"
        )
    },

    "chatbot_help": {
        "app_only": True,
        "keywords": [
            "what can the chatbot help me with",
            "what can you help with",
            "how can you help"
        ],
        "answer": (
            "🤖 **What the Chatbot Can Help You With**\n\n"
            "• Using the app and navigation\n"
            "• Understanding prediction tools\n"
            "• Viewing prediction history\n\n"
            "❌ I do not provide medical diagnosis or treatment."
        )
    },

    "limitations": {
        "app_only": True,
        "keywords": [
            "accuracy",
            "limitations",
            "is this reliable"
        ],
        "answer": (
            "⚠️ **Important Limitations**\n\n"
            "• This app does not replace a doctor\n"
            "• Predictions depend on the accuracy of input data\n"
            "• Results are for awareness only\n\n"
            "Always consult a healthcare professional."
        )
    }
}

def get_rule_based_answer(user_text, confidence_threshold=0.8):
    intents = classify_intents_with_confidence(user_text)

    if not intents:
        return None, 0.0

    answers = []
    confidences = []

    for intent, confidence in intents:
        if confidence >= confidence_threshold:
            answers.append(CHATBOT_KNOWLEDGE_BASE[intent]["answer"])
            confidences.append(confidence)

    if not answers:
        return None, 0.0

    combined_answer = "\n\n---\n\n".join(answers)
    avg_confidence = sum(confidences) / len(confidences)

    return combined_answer, round(avg_confidence, 2)


# -----------------------------------------------------
# Generic words to ignore during intent matching
# -----------------------------------------------------
GENERIC_WORDS = {
    "how", "use", "using",
    "prediction", "predict",
    "tool", "tools"
}

# =====================================================
# INTENT CLASSIFICATION WITH CONFIDENCE
# =====================================================
def classify_intents_with_confidence(user_text: str, max_intents=2):
    text = user_text.lower()
    words = set(text.split())

    intent_scores = []

    for intent, data in CHATBOT_KNOWLEDGE_BASE.items():
        score = 0.0
        priority = data.get("priority", 0)

        for kw in data["keywords"]:
            kw_lower = kw.lower()
            kw_words = set(kw_lower.split())

            # 1️⃣ Exact phrase match (strong)
            if kw_lower in text:
                score += 4.0
                continue

            # 2️⃣ Word overlap (ignore generic words)
            overlap = {
                w for w in kw_words.intersection(words)
                if w not in GENERIC_WORDS
            }
            score += 1.5 * len(overlap)

        # 3️⃣ Small priority boost
        score += 0.2 * priority

        if score > 0:
            intent_scores.append({
                "intent": intent,
                "score": score,
                "priority": priority
            })

    # Sort by score → priority
    intent_scores.sort(
        key=lambda x: (x["score"], x["priority"]),
        reverse=True
    )

    results = []
    for item in intent_scores[:max_intents]:
        confidence = min(item["score"] / 6.0, 1.0)
        results.append((item["intent"], round(confidence, 2)))

    return results

# =====================================================
# SAVE PREDICTION TO FIREBASE
# =====================================================
def save_prediction(user_id, disease, result, probability):
    if user_id is None:
        st.info("Please log in to save your prediction history.")
        return
    db.child("predictions").child(user_id).push({
        "disease": disease,
        "result": result,
        "probability": float(probability),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })
# =====================================================
# Generate PDF REPORT 
# =====================================================
def generate_disease_pdf(username, disease_name, df):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    # ================= PAGE 1: SUMMARY =================
    avg_risk = df["Risk %"].mean()
    max_risk = df["Risk %"].max()
    min_risk = df["Risk %"].min()
    total_preds = len(df)

    # Auto insight
    if avg_risk >= 70:
        insight = "Overall risk levels are consistently high. Immediate medical consultation is advised."
    elif avg_risk >= 30:
        insight = "Moderate risk levels detected. Regular monitoring is recommended."
    else:
        insight = "Overall risk levels are low. Maintain a healthy lifestyle."

    # ---------- SUMMARY HEADER ----------
    c.setFont("Helvetica-Bold", 18)
    c.drawString(50, height - 60, "DiagnoBridge – Summary Report")

    c.setFont("Helvetica", 11)
    c.drawString(50, height - 100, f"User: {username}")
    c.drawString(50, height - 120, f"Disease: {disease_name}")
    c.drawString(50, height - 140, f"Total Predictions: {total_preds}")
    c.drawString(50, height - 160, f"Average Risk: {avg_risk:.2f}%")
    c.drawString(50, height - 180, f"Highest Risk: {max_risk:.2f}%")
    c.drawString(50, height - 200, f"Lowest Risk: {min_risk:.2f}%")
    c.drawString(50, height - 220, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

    # ---------- AI INSIGHT ----------
    c.setFont("Helvetica-Oblique", 11)
    c.drawString(50, height - 260, "AI Health Insight:")
    c.drawString(50, height - 280, insight)

    c.setFont("Helvetica-Oblique", 9)
    c.drawString(
        50, 50,
        "Disclaimer: This report is generated by an AI model and is not a medical diagnosis."
    )

    # ✅ END OF SUMMARY PAGE
    c.showPage()

    # ================= PAGE 2+: DETAILED TABLE =================
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, height - 50, "Prediction History")

    # Table headers
    y = height - 90
    c.setFont("Helvetica-Bold", 10)
    c.drawString(50, y, "Result")
    c.drawString(130, y, "Risk %")
    c.drawString(200, y, "Timestamp")

    y -= 15
    c.setFont("Helvetica", 9)

    # Table rows
    for _, row in df.iterrows():
        c.drawString(50, y, row["result"])
        c.drawString(130, y, f'{row["Risk %"]}%')
        c.drawString(200, y, row["timestamp"].strftime("%Y-%m-%d %H:%M"))

        y -= 14
        if y < 60:
            c.showPage()
            y = height - 60
            c.setFont("Helvetica", 9)

    c.save()
    buffer.seek(0)
    return buffer
# ---------- DIABETES MODEL LOADER ----------
@st.cache_resource
def load_diabetes_model():
    artifacts = joblib.load("saved_mdls/diabetes_model.pkl")
    model = artifacts["model"]
    scaler = artifacts["scaler"]
    # ["Age","BMI","HighBP","HighChol","GlucoseProxy","RiskScore","SeverityScore","SeverityScore2"]
    features = artifacts["features"]
    return model, scaler, features

# ---------- HEART MODEL LOADER ----------
@st.cache_resource
def load_heart_model():
    artifacts = joblib.load("saved_mdls/heart_model.pkl")
    model = artifacts["model"]          # full Pipeline (preprocess + stacking)
    # ["Age","RestingBP","Cholesterol","MaxHR","Oldpeak","ChestPainType","ST_Slope","Thal"]
    features = artifacts["features"]
    return model, features

# ---------- PARKINSON'S MODEL LOADER ----------
@st.cache_resource
def load_parkinsons_model():
    artifacts = joblib.load("saved_mdls/parkinsons_model.pkl")
    model = artifacts["model"]
    scaler = artifacts["scaler"]
    # the 15 selected features
    features = artifacts["features"]
    return model, scaler, features

# ---------- SYMPTOM SVM MODEL LOADER ----------
@st.cache_resource
def load_symptom_model():
    artifacts = joblib.load("saved_mdls/symptom_model.pkl")
    model = artifacts["model"]              # SVM
    symptoms = artifacts["symptoms"]        # list of 132 symptom names
    label_encoder = artifacts["label_encoder"]
    return model, symptoms, label_encoder

# ---------- BERT SYMPTOM NLP MODEL LOADER ----------
@st.cache_resource
def load_bert_symptom_model():
    base_dir = "saved_mdls/bert_symptom_nlp"
    tokenizer = DistilBertTokenizerFast.from_pretrained(base_dir)
    model = DistilBertForSequenceClassification.from_pretrained(base_dir)
    le_art = joblib.load(os.path.join(base_dir, "label_encoder.pkl"))
    label_encoder = le_art["label_encoder"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    return tokenizer, model, label_encoder, device

# ---------- MEDICAL TIPS HELPERS ----------

def show_diabetes_tips(high_risk: bool):
    st.markdown("### 🩺 Diabetes – General Health Tips")
    if high_risk:
        st.markdown(
            """
- Schedule a **consultation with a doctor** for proper evaluation and lab tests (FBS, PPBS, HbA1c).
- Try to **limit sugary drinks, sweets, and refined carbohydrates** in your daily diet.
- Include more **vegetables, whole grains, and fibre-rich foods**.
- Aim for at least **30 minutes of walking or light exercise** on most days.
- If you are overweight, work on **gradual weight reduction** under medical guidance.
- Avoid **smoking and excessive alcohol**, as they increase diabetes-related complications.
            """
        )
    else:
        st.markdown(
            """
- Continue following a **balanced diet** with limited sugar and refined foods.
- Maintain **regular physical activity** (walking, cycling, light workouts).
- Go for **periodic health check-ups** (blood sugar, BP, cholesterol).
- Keep your **weight, BP, and cholesterol** within recommended ranges.
- Avoid smoking and manage **stress and sleep** properly.
            """
        )

def show_heart_tips(high_risk: bool):
    st.markdown("### ❤️ Heart Health Tips")
    if high_risk:
        st.markdown(
            """
- Consult a **cardiologist / doctor** as early as possible for further tests (ECG, echo, stress test).
- **Avoid heavy exertion** until your condition is properly evaluated.
- Reduce **salt, oily and fried foods**, and processed/fast foods.
- Focus on a **heart-healthy diet**: fruits, vegetables, whole grains, lean protein.
- If you smoke, take steps to **stop smoking**; avoid second-hand smoke too.
- Monitor **blood pressure, blood sugar, and cholesterol** regularly.
            """
        )
    else:
        st.markdown(
            """
- Maintain a **heart-healthy lifestyle**: exercise, balanced diet, proper sleep.
- Keep **BP, cholesterol, and blood sugar** under regular check.
- Limit **salt intake, deep-fried foods, and junk food**.
- Avoid smoking and reduce **alcohol and stress**.
- Continue routine **health check-ups** as advised by your doctor.
            """
        )

def show_parkinsons_tips(high_risk: bool):
    st.markdown("### 🧠 Parkinson’s – Supportive Tips")
    if high_risk:
        st.markdown(
            """
- Meet a **neurologist** for detailed assessment and confirmation.
- Note any symptoms like **tremor, stiffness, slowness, or imbalance** and share them with your doctor.
- Gentle **stretching, walking, and balance exercises** may help – only under professional guidance.
- Maintain a **regular sleep routine** and try to reduce stress.
- Consider **speech or physiotherapy** if recommended by your doctor.
            """
        )
    else:
        st.markdown(
            """
- If you notice persistent tremor, stiffness, or balance issues, **consult a doctor early**.
- Keep yourself **physically active** with safe exercises like walking and stretching.
- Maintain a healthy lifestyle with **good sleep, nutrition, and stress management**.
- Go for regular **check-ups** if you have any neurological concerns.
            """
        )

# ---------- BERT PREDICTION HELPER ----------
def predict_disease_from_text(text: str,
                              tokenizer,
                              model,
                              label_encoder,
                              device: str = "cpu") -> str:
    """Run DistilBERT on symptom text and return disease name."""
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits
        pred_id = int(torch.argmax(logits, dim=-1).cpu().numpy()[0])

    disease = label_encoder.inverse_transform([pred_id])[0]
    return disease

# =====================================================
# CHATBOT LOGIC (APP GUIDE + USER DATA + GENERAL INFO)
# =====================================================

def app_guide_reply(text):
    text = text.lower()

    # ---- General app usage ----
    if any(k in text for k in [
        "how do i use",
        "how to use",
        "use this app",
        "use diagnobridge",
        "getting started"
    ]):
        return (
            "📌 **How to Use DiagnoBridge**\n\n"
            "1️⃣ Select a prediction tool from the **left sidebar**\n"
            "2️⃣ Enter the required health values or symptoms\n"
            "3️⃣ Click **Predict**\n"
            "4️⃣ View risk result and medical tips\n"
            "5️⃣ Visit **Prediction History** to track past results"
        )

    # ---- Disease-specific prediction usage ----
    if any(k in text for k in [
        "use disease prediction",
        "use prediction tool",
        "how do i use disease",
        "how to predict disease",
        "how does disease prediction work"
    ]):
        return (
            "🧪 **Using Disease Prediction Tools**\n\n"
            "• Choose a disease (Diabetes, Heart, Parkinson’s) from the sidebar\n"
            "• Enter medical values shown on the screen\n"
            "• Click **Predict** to see risk percentage\n"
            "• Open **Medical Tips** for guidance\n\n"
            "⚠️ This is a prediction, not a diagnosis."
        )

    # ---- Symptom-based prediction ----
    if any(k in text for k in [
        "symptom prediction",
        "predict from symptoms",
        "only symptoms",
        "symptom based"
    ]):
        return (
            "🧾 **Symptom-based Prediction**\n\n"
            "You have two options:\n"
            "1️⃣ **Checklist Mode** – select symptoms manually\n"
            "2️⃣ **Free-text Mode** – describe symptoms in your own words\n\n"
            "The system predicts the most likely disease based on patterns."
        )

    # ---- Prediction history navigation ----
    if any(k in text for k in [
        "prediction history",
        "past predictions",
        "saved results",
        "history tab"
    ]):
        return (
            "📊 **Prediction History**\n\n"
            "• View all past predictions\n"
            "• Track risk trends over time\n"
            "• Download PDF health reports\n\n"
            "Access it from the **Prediction History** tab at the top."
        )

    return ""

def history_based_reply(text, user_id):
    if user_id is None:
        return ""

    records = db.child("predictions").child(user_id).get()
    if not records.each():
        return ""

    df = pd.DataFrame([r.val() for r in records.each()])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["Risk %"] = (df["probability"] * 100).round(2)

    text = text.lower()

    if "last prediction" in text or "latest prediction" in text:
        latest = df.sort_values("timestamp", ascending=False).iloc[0]
        return (
            f"📌 **Your Latest Prediction**\n\n"
            f"- Disease: **{latest['disease']}**\n"
            f"- Result: **{latest['result']}**\n"
            f"- Risk: **{latest['Risk %']}%**"
        )

    if "how many predictions" in text:
        return f"📊 You have made **{len(df)} predictions** so far."

    if "highest risk" in text:
        high = df.loc[df["Risk %"].idxmax()]
        return (
            f"⚠️ **Highest Risk Recorded**\n\n"
            f"- Disease: **{high['disease']}**\n"
            f"- Risk: **{high['Risk %']}%**"
        )

    return ""

 
def api_medical_reply(text):
    return (
        "🩺 **General Medical Information**\n\n"
        "I can explain disease definitions, causes, and prevention.\n"
        "For diagnosis or treatment, please consult a medical professional."
    )

def farewell_reply(text):
    text = text.lower().strip()

    if any(k in text for k in [
        "bye", "goodbye", "see you",
        "thanks", "thank you",
        "ok", "okay", "cool",
        "done", "exit", "quit"
    ]):
        return (
            "👋 **Got it!**\n\n"
            "Thanks for using **DiagnoBridge**.\n"
            "If you need help again, just come back anytime 💙"
        )

    return None

def chatbot_router(user_text):

    # 0️⃣ Farewell / end conversation
    bye = farewell_reply(user_text)
    if bye:
        st.session_state.chat_ended = True
        return bye

    user_id = st.session_state.get("user")

    # Build short chat context (last 5 messages)
    chat_context = ""
    for role, msg in st.session_state.chat_history[-5:]:
        chat_context += f"{role}: {msg}\n"

    # --------------------------------------------------
    # 1️⃣ USER-SPECIFIC HISTORY QUESTIONS (SAFE)
    # --------------------------------------------------
    history_reply = history_based_reply(user_text, user_id)
    if history_reply:
        return history_reply

    # --------------------------------------------------
    # 2️⃣ RULE-BASED APP QUESTIONS (HIGH CONFIDENCE ONLY)
    # --------------------------------------------------
    rule_answer, confidence = get_rule_based_answer(user_text)

    if rule_answer and confidence >= 0.8:
        return rule_answer + f"\n\n🧠 *Confidence: {confidence:.0%}*"

    # --------------------------------------------------
    # 3️⃣ DEFAULT → OLLAMA (GENERAL HEALTH / UNKNOWN)
    # --------------------------------------------------
    return ollama_medical_reply(user_text, chat_context)

def get_followup_suggestions(user_text):
    text = user_text.lower()

    if any(k in text for k in ["diabetes", "heart", "parkinson"]):
        return [
            "How is risk calculated?",
            "What does high risk mean?",
            "Where can I see my prediction history?"
        ]

    if any(k in text for k in ["use", "prediction", "tool"]):
        return [
            "Which prediction should I use?",
            "How do I enter symptoms?",
            "Where are my saved results?"
        ]

    if "history" in text:
        return [
            "Which disease has highest risk?",
            "Can I download my report?",
            "How many predictions have I made?"
        ]

    return [
        "How do I use this app?",
        "What is diabetes?",
        "Show my prediction history"
    ]

def get_contextual_suggestions():
    tab = st.session_state.get("active_tab", "chatbot")

    if tab == "prediction":
        return [
            "How is this risk calculated?",
            "What does high risk mean?",
            "Is this a medical diagnosis?"
        ]

    if tab == "history":
        return [
            "Show my latest prediction",
            "Which disease has highest risk?",
            "Can I download my report?"
        ]

    return [
        "How do I use this app?",
        "What predictions are available?",
        "What can you help me with?"
    ]

# ---------- DIABETES UI ----------
def diabetes_page():
    st.write("Provide the following details to estimate diabetes risk:")

    model, scaler, features = load_diabetes_model()
    user_id = st.session_state.get("user")
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age (years)", min_value=1, max_value=120, value=40, step=1)
        bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0, step=0.1)
        glucose = st.number_input("Glucose level (mg/dL)", min_value=50.0, max_value=300.0, value=120.0, step=1.0)

    with col2:
        highbp = st.selectbox(
            "High Blood Pressure diagnosed?",
            options=[0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No"
        )
        highchol = st.selectbox(
            "High Cholesterol diagnosed?",
            options=[0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No"
        )

    if st.button("🔍 Predict Diabetes Risk"):
        # Recreate engineered features EXACTLY like in training
        risk_score = 0.3 * bmi + 0.3 * glucose + 5 * highbp + 4 * highchol + 0.1 * age
        severity = 0.4 * risk_score + 0.2 * bmi + 0.2 * glucose + 0.2 * highbp
        severity2 = 0.5 * glucose + 0.3 * risk_score + 0.1 * bmi + 0.1 * highbp

        row = pd.DataFrame([{
            "Age": age,
            "BMI": bmi,
            "HighBP": highbp,
            "HighChol": highchol,
            "GlucoseProxy": glucose,
            "RiskScore": risk_score,
            "SeverityScore": severity,
            "SeverityScore2": severity2,
        }])

        X_scaled = scaler.transform(row[features])
        pred = model.predict(X_scaled)[0]
        proba = model.predict_proba(X_scaled)[0, 1]
        
        save_prediction(
            user_id=user_id,
            disease="Diabetes",
            result="High Risk" if pred == 1 else "Low Risk",
            probability=proba
        )

        st.markdown("---")
        if pred == 1:
            st.error(f"**High diabetes risk detected. Probability: {proba:.2%}**")
            high_risk = True
        else:
            st.success(f"**Low diabetes risk. Probability of diabetes: {proba:.2%}**")
            high_risk = False

        with st.expander("📘 Medical Tips"):
            show_diabetes_tips(high_risk)

        st.caption("Note: This is a prediction model and not a medical diagnosis.")

# ---------- HEART DISEASE UI ----------
def heart_page():
    model, features = load_heart_model()

    user_id = st.session_state.get("user")

    st.write("Provide the following details to estimate heart disease risk:")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age (years)", min_value=1, max_value=120, value=50, step=1)
        resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", min_value=70, max_value=250, value=130, step=1)
        cholesterol = st.number_input("Cholesterol (mg/dL)", min_value=100, max_value=600, value=240, step=1)
        max_hr = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=250, value=150, step=1)

    with col2:
        oldpeak = st.number_input("Oldpeak (ST depression)", min_value=-2.0, max_value=10.0, value=1.0, step=0.1)

        chest_pain = st.selectbox(
            "Chest Pain Type",
            options=["ASY", "NAP", "ATA", "TA"],
            format_func=lambda x: {
                "ASY": "ASY – Asymptomatic",
                "NAP": "NAP – Non-anginal pain",
                "ATA": "ATA – Atypical angina",
                "TA":  "TA – Typical angina",
            }.get(x, x)
        )

        st_slope = st.selectbox(
            "ST Slope",
            options=["Up", "Flat", "Down"],
            format_func=lambda x: f"ST Slope: {x}"
        )

        thal = st.selectbox(
            "Thal (Thalassemia)",
            options=["normal", "fixed", "reversible", "unknown"],
            format_func=lambda x: f"Thal: {x}"
        )

    if st.button("🔍 Predict Heart Disease Risk"):
        row = pd.DataFrame([{
            "Age": age,
            "RestingBP": resting_bp,
            "Cholesterol": cholesterol,
            "MaxHR": max_hr,
            "Oldpeak": oldpeak,
            "ChestPainType": chest_pain,
            "ST_Slope": st_slope,
            "Thal": thal,
        }])

        # Full pipeline handles preprocessing inside (scaling + one-hot)
        pred = model.predict(row)[0]
        proba = model.predict_proba(row)[0, 1]

        save_prediction(
            user_id=user_id,
            disease="Heart Disease",
            result="High Risk" if pred == 1 else "Low Risk",
            probability=proba
        )

        st.markdown("---")
        if pred == 1:
            st.error(f"**High risk of heart disease. Probability: {proba:.2%}**")
            high_risk = True
        else:
            st.success(f"**Low risk of heart disease. Probability of heart disease: {proba:.2%}**")
            high_risk = False

        with st.expander("📘 Medical Tips"):
            show_heart_tips(high_risk)

        st.caption("Note: This is a prediction model and not a medical diagnosis.")

# ---------- PARKINSON'S UI ----------
def parkinsons_page():
    st.write("Provide the following voice-related features to estimate Parkinson’s risk:")

    model, scaler, features = load_parkinsons_model()

    user_id = st.session_state.get("user")

    st.markdown("### Enter voice feature values")

    # Pre-filled default values
    default_values = {
        "MDVP:Fo(Hz)": 120.0,
        "MDVP:Fhi(Hz)": 150.0,
        "MDVP:Flo(Hz)": 90.0,
        "MDVP:Jitter(%)": 0.005,
        "MDVP:Jitter(Abs)": 0.00004,
        "MDVP:RAP": 0.003,
        "MDVP:PPQ": 0.0035,
        "Jitter:DDP": 0.009,
        "MDVP:Shimmer": 0.03,
        "MDVP:Shimmer(dB)": 0.25,
        "Shimmer:APQ5": 0.02,
        "MDVP:APQ": 0.02,
        "NHR": 0.02,
        "HNR": 22.0,
        "PPE": 0.25
    }

    # Tooltips for each feature
    feature_tooltips = {
        "MDVP:Fo(Hz)": "Average pitch of the voice.",
        "MDVP:Fhi(Hz)": "Highest pitch reached during speech.",
        "MDVP:Flo(Hz)": "Lowest pitch reached during speech.",
        "MDVP:Jitter(%)": "Percentage variability in frequency — higher means more instability.",
        "MDVP:Jitter(Abs)": "Absolute jitter value indicating voice frequency variation.",
        "MDVP:RAP": "Short-term pitch variability measure.",
        "MDVP:PPQ": "Pitch variation over several cycles.",
        "Jitter:DDP": "Advanced jitter measurement.",
        "MDVP:Shimmer": "Amplitude variation — higher shimmer indicates shaky voice.",
        "MDVP:Shimmer(dB)": "Shimmer measured in decibels.",
        "Shimmer:APQ5": "Amplitude perturbation across 5 cycles.",
        "MDVP:APQ": "Amplitude perturbation across 11 cycles.",
        "NHR": "Amount of noise in the voice signal.",
        "HNR": "Ratio of harmonic sound to noise — lower means more hoarseness.",
        "PPE": "Irregularity measure of pitch stability."
    }

    cols = st.columns(3)
    inputs = {}

    for idx, feature in enumerate(features):
        with cols[idx % 3]:
            inputs[feature] = st.number_input(
                feature,
                value=float(default_values[feature]),
                step=0.0001,
                format="%.5f",
                help=feature_tooltips[feature]
            )

    if st.button("🔍 Predict Parkinson's Risk"):
        row = pd.DataFrame([inputs])

        X_scaled = scaler.transform(row[features])
        pred = model.predict(X_scaled)[0]
        proba = model.predict_proba(X_scaled)[0, 1]

        save_prediction(
            user_id=user_id,
            disease="Parkinson's Disease",
            result="High Risk" if pred == 1 else "Low Risk",
            probability=proba
        )

        st.markdown("---")
        if pred == 1:
            st.error(f"**High risk of Parkinson’s disease. Probability: {proba:.2%}**")
            high_risk = True
        else:
            st.success(f"**Low risk of Parkinson’s disease. Probability: {proba:.2%}**")
            high_risk = False

        with st.expander("📘 Medical Tips"):
            show_parkinsons_tips(high_risk)

        st.caption("Note: This is a prediction model and not a medical diagnosis.")

# ---------- SYMPTOM-BASED PREDICTION UI ----------
def symptom_page():
    st.subheader("🧾 Symptom-based Disease Prediction")
    st.write("Use either a symptom checklist or free-text description.")

    tab1, tab2 = st.tabs(["✅ Checklist Mode (SVM)", "💬 Free-text (BERT NLP)"])

    # --------- TAB 1: CHECKLIST MODE (SVM ON 132 SYMPTOMS) ---------
    with tab1:
        st.markdown("### Select your symptoms")

        svm_model, symptom_list, label_encoder = load_symptom_model()

        selected_symptoms = st.multiselect(
            "Choose the symptoms you are experiencing:",
            options=sorted(symptom_list),
            help="Start typing to search symptoms."
        )

        if st.button("🔍 Predict Disease (Checklist)", key="btn_checklist"):
            if not selected_symptoms:
                st.warning("Please select at least one symptom.")
            else:
                # Build 0/1 vector
                row = {sym: 0 for sym in symptom_list}
                for s in selected_symptoms:
                    row[s] = 1
                X_input = pd.DataFrame([row])

                # Predict
                preds = svm_model.predict(X_input)
                probs = svm_model.predict_proba(X_input)[0]

                # Map probs indices → disease names using classes_ alignment
                class_ids = svm_model.classes_
                disease_names = label_encoder.inverse_transform(class_ids)
                sorted_idx = np.argsort(probs)[::-1]

                top_idx = sorted_idx[0]
                top_disease = disease_names[top_idx]
                top_prob = probs[top_idx]

                st.markdown("---")
                st.markdown("### 🎯 Predicted Disease (Checklist Model)")
                st.write(f"**Most likely disease:** `{top_disease}`")
                st.write(f"**Confidence:** `{top_prob:.2%}`")

                st.markdown("#### Top 3 possible diseases:")
                for rank, idx in enumerate(sorted_idx[:3], start=1):
                    st.write(f"{rank}. `{disease_names[idx]}` — `{probs[idx]:.2%}`")

                st.caption("Note: This is a prediction model and not a medical diagnosis.")

    # --------- TAB 2: FREE-TEXT MODE (BERT NLP) ---------
    with tab2:
        st.markdown("### Describe your symptoms in your own words")

        tokenizer, bert_model, label_encoder, device = load_bert_symptom_model()

        symptoms_text = st.text_area(
            "Symptom description",
            placeholder="Example: fever, headache, body pain, vomiting for 2 days",
            height=120,
        )

        if st.button("🔍 Predict Disease (BERT NLP)", key="btn_bert"):
            if not symptoms_text.strip():
                st.warning("Please enter some symptoms.")
            else:
                disease = predict_disease_from_text(
                    symptoms_text, tokenizer, bert_model, label_encoder, device
                )

                st.markdown("---")
                st.markdown("### 🤖 NLP-based Prediction (DistilBERT)")
                st.write(f"**Most likely disease:** `{disease}`")
                st.caption(
                    "This model reads your symptom text using DistilBERT. "
                    "It is trained on the same curated symptom–disease dataset as the checklist model."
                )
                st.caption("Note: This is a prediction model and not a medical diagnosis.")

# PREDICTION HISTORY DASHBOARD
# =====================================================
def prediction_history_page():
    st.subheader("📊 Prediction History Dashboard")
    st.caption("Track your past predictions and monitor health risk trends over time.")

    user_id = st.session_state.get("user")
    if user_id is None:
        st.info("Please log in to view your prediction history.")
        return

    records = db.child("predictions").child(user_id).get()
    if not records.each():
        st.info("No prediction history available yet.")
        return

    # ---------------- LOAD DATA ----------------
    df = pd.DataFrame([r.val() for r in records.each()])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["Risk %"] = (df["probability"] * 100).round(2)
    df.drop(columns=["probability"], inplace=True)
    df.sort_values("timestamp", ascending=False, inplace=True)

    # ---------------- FILTERS ----------------
    st.markdown("### 🧬 Filters")

    col1, col2 = st.columns(2)

    with col1:
        disease_filter = st.selectbox(
            "Filter by disease",
            ["All"] + sorted(df["disease"].unique().tolist())
        )

    with col2:
        start_date, end_date = st.date_input(
            "Date range",
            [df["timestamp"].min().date(), df["timestamp"].max().date()]
        )

    filtered_df = df.copy()

    if disease_filter != "All":
        filtered_df = filtered_df[filtered_df["disease"] == disease_filter]

    filtered_df = filtered_df[
        (filtered_df["timestamp"].dt.date >= start_date) &
        (filtered_df["timestamp"].dt.date <= end_date)
    ]

    if filtered_df.empty:
        st.warning("No records for selected filters.")
        return

    # ======================================================
    # 📌 SUMMARY STATISTICS
    # ======================================================
    st.markdown("### 📌 Summary Statistics")

    total_preds = len(filtered_df)
    avg_risk = filtered_df["Risk %"].mean()
    max_risk = filtered_df["Risk %"].max()
    min_risk = filtered_df["Risk %"].min()
    latest = filtered_df.sort_values("timestamp", ascending=False).iloc[0]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Predictions", total_preds)
    c2.metric("Average Risk", f"{avg_risk:.2f}%")
    c3.metric("Highest Risk", f"{max_risk:.2f}%")
    c4.metric("Lowest Risk", f"{min_risk:.2f}%")

    st.markdown(
        f"""
**Latest Prediction**
- Disease: **{latest['disease']}**
- Result: **{latest['result']}**
- Risk: **{latest['Risk %']}%**
- Time: **{latest['timestamp'].strftime('%Y-%m-%d %H:%M')}**
"""
    )

    # ======================================================
    # 🧠 AUTO-GENERATED HEALTH INSIGHTS
    # ======================================================
    st.markdown("### 🧠 AI Health Insights")

    high_risk_count = (filtered_df["Risk %"] >= 70).sum()
    low_risk_count = (filtered_df["Risk %"] < 30).sum()

    if avg_risk >= 70:
        insight = "Overall risk levels are consistently high. Immediate medical consultation is recommended."
    elif avg_risk >= 30:
        insight = "Risk levels fluctuate between moderate and high. Continuous monitoring is advised."
    else:
        insight = "Most predictions indicate low risk. Maintaining a healthy lifestyle is recommended."

    st.info(
        f"""
- {insight}
- High-risk predictions: **{high_risk_count}**
- Low-risk predictions: **{low_risk_count}**
"""
    )

    # ---------------- RESET PAGE ON FILTER CHANGE ----------------
    filter_signature = (disease_filter, start_date, end_date)
    if st.session_state.get("last_filter") != filter_signature:
        st.session_state.page_num = 1
        st.session_state.last_filter = filter_signature

    # ---------------- PAGINATION ----------------
    ROWS_PER_PAGE = 6
    total_pages = max(1, (len(filtered_df) - 1) // ROWS_PER_PAGE + 1)

    if "page_num" not in st.session_state:
        st.session_state.page_num = 1

    st.session_state.page_num = min(
        max(1, st.session_state.page_num),
        total_pages
    )

    st.markdown("### 📋 Prediction Records")

    col_prev, col_jump, col_next = st.columns([1, 2, 1])

    with col_prev:
        if st.button("⬅️ Prev", disabled=st.session_state.page_num == 1):
            st.session_state.page_num -= 1
            st.rerun()

    with col_jump:
        selected_page = st.selectbox(
            "Page",
            list(range(1, total_pages + 1)),
            index=st.session_state.page_num - 1
        )
        if selected_page != st.session_state.page_num:
            st.session_state.page_num = selected_page
            st.rerun()

    with col_next:
        if st.button("Next ➡️", disabled=st.session_state.page_num == total_pages):
            st.session_state.page_num += 1
            st.rerun()

    start_idx = (st.session_state.page_num - 1) * ROWS_PER_PAGE
    end_idx = start_idx + ROWS_PER_PAGE
    page_df = filtered_df.iloc[start_idx:end_idx].copy()

    def color_risk(val):
        if val < 30:
            return "background-color: #d4edda"
        elif val < 70:
            return "background-color: #fff3cd"
        else:
            return "background-color: #f8d7da"

    st.dataframe(
        page_df.style.applymap(color_risk, subset=["Risk %"]),
        use_container_width=True
    )

    # ======================================================
    # 📄 DOWNLOAD PDF REPORTS (PER DISEASE)
    # ======================================================
    with st.expander("📄 Download PDF Reports", expanded=False):

        username = st.session_state.get("username", "User")

        if disease_filter == "All":
            for disease in sorted(filtered_df["disease"].unique()):
                disease_df = filtered_df[filtered_df["disease"] == disease]
                if disease_df.empty:
                    continue

                pdf_buffer = generate_disease_pdf(username, disease, disease_df)

                st.download_button(
                    label=f"⬇️ Download {disease} Report",
                    data=pdf_buffer,
                    file_name=f"DiagnoBridge_{disease.replace(' ', '_')}_Report.pdf",
                    mime="application/pdf",
                    key=f"pdf_{disease}"
                )
        else:
            pdf_buffer = generate_disease_pdf(
                username, disease_filter, filtered_df
            )

            st.download_button(
                label=f"⬇️ Download {disease_filter} Report",
                data=pdf_buffer,
                file_name=f"DiagnoBridge_{disease_filter.replace(' ', '_')}_Report.pdf",
                mime="application/pdf",
                key=f"pdf_{disease_filter}"
            )

    st.markdown("</div>", unsafe_allow_html=True)

    # ======================================================
    # 📊 RISK TRENDS (FILTER + PAGE AWARE)
    # ======================================================
    st.markdown("### 📊 Risk Trend by Disease")

    # 🔑 Decide chart source
    chart_df = filtered_df if disease_filter == "All" else page_df

    tab_labels = []
    if disease_filter == "All":
        tab_labels.append("All Diseases")

    disease_list = sorted(chart_df["disease"].unique().tolist())
    tab_labels.extend(disease_list)

    tabs = st.tabs(tab_labels)
    tab_idx = 0

    # --------- ALL DISEASES (ALL PAGES) ---------
    if disease_filter == "All":
        with tabs[0]:
            st.markdown("#### All Diseases Risk Trend")

            safe_df = chart_df.copy()
            safe_df["disease_safe"] = (
                safe_df["disease"]
                .str.replace("'", "", regex=False)
                .str.replace(" ", "_", regex=False)
            )

            pivot_df = (
                safe_df
                .sort_values("timestamp")
                .pivot_table(
                    index="timestamp",
                    columns="disease_safe",
                    values="Risk %",
                    aggfunc="mean"
                )
                .ffill()
            )

            if pivot_df.shape[0] < 2:
                st.info("Not enough data to display trends.")
            else:
                st.line_chart(pivot_df)

        tab_idx = 1

    # --------- PER-DISEASE TABS ---------
    for disease in disease_list:
        with tabs[tab_idx]:
            st.markdown(f"#### {disease} Risk Trend")

            disease_df = (
                chart_df[chart_df["disease"] == disease]
                .sort_values("timestamp")
                .set_index("timestamp")
            )

            if len(disease_df) < 2:
                st.info("Not enough data points to display trend.")
            else:
                st.line_chart(disease_df["Risk %"])

        tab_idx += 1

def health_chatbot_page():

    # ---------- SESSION STATE INIT ----------
    if "chat_started" not in st.session_state:
        st.session_state.chat_started = False

    if "chat_ended" not in st.session_state:
        st.session_state.chat_ended = False

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "pending_question" not in st.session_state:
        st.session_state.pending_question = None

    # ---------- HEADER ----------
    st.subheader("🤖 Health Chatbot")
    st.caption("I help you use the app, understand your data, and answer basic health questions.")

    # ---------- CLEAR CHAT ----------
    col1, col2 = st.columns([6, 1])
    with col2:
        if st.button("🧹 Clear Chat"):
            st.session_state.chat_history = []
            st.session_state.chat_started = False
            st.session_state.chat_ended = False
            st.session_state.pending_question = None
            st.rerun()

    # ---------- QUICK HELP (ONLY BEFORE CHAT STARTS) ----------
    if not st.session_state.chat_started:
        st.info(
            "💡 **You can ask me about:**\n"
            "- How to use this website\n"
            "- Navigation, tabs, and tools\n"
            "- Viewing prediction history\n"
            "- Downloading PDF reports\n"
            "- General health education (definitions only)"
        )

        st.markdown("### 🧭 Try one of these questions:")
        cols = st.columns(2)
        for i, q in enumerate(CHATBOT_SUGGESTED_QUESTIONS):
            if cols[i % 2].button(q, key=f"start_q_{i}"):
                st.session_state.pending_question = q
                st.session_state.chat_started = True
                st.rerun()
        st.markdown("---")

    # ---------- SHOW CHAT HISTORY ----------
    for role, msg in st.session_state.chat_history:
        with st.chat_message(role):
            st.markdown(msg)

    # ---------- CHAT INPUT ----------
    user_input = st.chat_input("Ask me about the app or your health data...")

    # Handle suggested question click
    if st.session_state.pending_question:
        user_input = st.session_state.pending_question
        st.session_state.pending_question = None

    # ---------- PROCESS MESSAGE ----------
    if user_input:
        st.session_state.chat_started = True

        # Save user message
        st.session_state.chat_history.append(("user", user_input))
        with st.chat_message("user"):
            st.markdown(user_input)

        # ---------- FAREWELL HANDLING ----------
        if user_input.lower().strip() in ["bye", "goodbye", "exit", "thanks", "thank you"]:
            farewell = "👋 Goodbye! I’ve cleared the chat. Take care."
            st.session_state.chat_history.append(("assistant", farewell))
            with st.chat_message("assistant"):
                st.markdown(farewell)

            st.session_state.chat_ended = True
            time.sleep(3)

            st.session_state.chat_history = []
            st.session_state.chat_started = False
            st.session_state.chat_ended = False
            st.session_state.pending_tab_switch = None
            st.rerun()
            return

        # ---------- NORMAL BOT RESPONSE ----------
        response = chatbot_router(user_input)

        # Follow-ups only if not ended
        followups = get_followup_suggestions(user_input)
        if followups:
            response += "\n\n💡 **You can also ask:**\n"
            for q in followups:
                response += f"- {q}\n"

        st.session_state.chat_history.append(("assistant", response))
        with st.chat_message("assistant"):
            st.markdown(response)

# ---------- MAIN APP ----------
def run_app():
    # ---------------- SESSION INIT ----------------
    if "active_tab" not in st.session_state:
        st.session_state.active_tab = "prediction"  # default page

    username = st.session_state.get("username", "User")

    # ===================== SIDEBAR =====================
    st.sidebar.markdown("## ⚙️ Navigation")

    tool_options = [
        "🩺 Diabetes Prediction",
        "❤️ Heart Disease Prediction",
        "🧠 Parkinsons Prediction",
        "Symptom-based Prediction"
    ]

    choice = st.sidebar.selectbox(
        "Choose a prediction tool:",
        tool_options
    )

    # Logout section
    st.sidebar.markdown("---")
    with st.sidebar.expander("👤 Account", expanded=True):
        st.markdown(f"**Logged in as:** `{username}`")
        if st.button("🚪 Logout"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.query_params.clear()
            st.session_state["user"] = None
            st.session_state["username"] = ""
            st.success("You have been logged out. Returning to login...")
            time.sleep(1)
            st.rerun()

    # ===================== HEADER =====================
    st.markdown(
        "<h2 style='text-align:center;'>DiagnoBridge Dashboard</h2>",
        unsafe_allow_html=True
    )
    st.markdown(
        f"<h4 style='text-align:center;'>👋 Welcome, {username}</h4>",
        unsafe_allow_html=True
    )
    st.markdown("---")

    # ===================== TOP NAV BUTTONS =====================
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("🩺 Prediction Tool"):
            st.session_state.active_tab = "prediction"
            st.rerun()

    with col2:
        if st.button("📊 Prediction History"):
            st.session_state.active_tab = "history"
            st.rerun()

    with col3:
        if st.button("🤖 Health Chatbot"):
            st.session_state.active_tab = "chatbot"
            st.rerun()

    st.markdown("---")

    # ===================== CONDITIONAL NAVIGATION =====================
    active_tab = st.session_state.active_tab

    # ---------- PREDICTION PAGE ----------
    if active_tab == "prediction":
        st.markdown(f"### {choice}")
        st.markdown("---")

        if choice == "🩺 Diabetes Prediction":
            diabetes_page()

        elif choice == "❤️ Heart Disease Prediction":
            heart_page()

        elif choice == "🧠 Parkinsons Prediction":
            parkinsons_page()

        elif choice == "Symptom-based Prediction":
            symptom_page()

    # ---------- HISTORY PAGE ----------
    elif active_tab == "history":
        prediction_history_page()

    # ---------- CHATBOT PAGE ----------
    elif active_tab == "chatbot":
        health_chatbot_page()

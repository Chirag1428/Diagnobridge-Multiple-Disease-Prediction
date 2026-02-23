# 🏥 DiagnoBridge – AI Powered Smart Healthcare Platform

## 📌 Overview

**DiagnoBridge** is an AI-driven **multi-disease prediction platform** designed to support preventive healthcare and early risk detection.

The system **integrates** machine learning, NLP-based symptom analysis, secure authentication, personalized dashboards, PDF report generation, and a **hybrid health chatbot** — delivering a complete smart healthcare experience through a **unified web application**.

---

## 🎯 Business Goals

* Enable early detection of Diabetes, Heart Disease, and Parkinson’s

* Provide both clinical parameter-based and symptom-based prediction

* Improve prediction accuracy using ensemble learning

* Offer post-prediction medical guidance

* Track patient risk trends over time

* Enhance user engagement through conversational assistance

* Create a unified, secure, and user-centric healthcare platform

---

## 🧠 Core Modules

# 🔹 Disease Prediction (Clinical Parameters)

* Stacking Ensemble (Random Forest + XGBoost + Logistic Regression)

* XGBoost for enhanced Parkinson’s prediction

* Probability-based risk output with medical tips


# 🔹 Symptom-Based Prediction

* Checklist-based prediction (SVM)

* Free-text symptom analysis using DistilBERT (NLP)

* Ranked disease probabilities


# 🔹 Prediction History Dashboard

* Stores results securely in Firebase

* Displays:

  -> Average risk

  -> Highest risk

  -> Lowest risk

* Disease-wise trend visualization

* Downloadable PDF health reports


# 🔹 Health Chatbot

* Hybrid model:

  -> Rule-based responses

  -> Local AI-powered conversational support

* Provides:

  -> Application guidance

  -> General health education

  -> Safe, non-diagnostic replies

---

## 📊 Model Performance

| Disease       | Model                    | Accuracy |
| ------------- | ------------------------ | -------- |
| Diabetes      | Stacking (RF + XGB + LR) | 87.39%   |
| Heart Disease | Stacking (RF + XGB + LR) | 94.33%   |
| Parkinson’s   | XGBoost                  | 92.31%   |
| Symptom-Based | XGBoost                  | 97.61%   |

---

## 🏗 System Architecture

* Streamlit Web Application

* ML Model Layer (Stacking, XGBoost, SVM, DistilBERT)

* Firebase (Authentication + Data Storage)

* PDF Report Engine

* Hybrid AI Chatbot

---

## ⚙ Features

* Secure Firebase login

* Multi-disease prediction

* Dual-mode symptom input (Checklist + NLP)

* Real-time risk probability

* Instant medical tips

* Prediction history tracking

* PDF export functionality

* Interactive chatbot

* Preventive healthcare focus

---

## 🛠 Built With

* Python with Matplotlib / Visualization Tools

* Streamlit and Scikit-learn

* XGBoost, SVM and DistilBERT (Transformers)

* Firebase

---

## 🚀 Innovation Highlights

* Integrated multi-disease prediction in one platform

* Combines structured clinical data + unstructured symptom text

* Post-prediction medical advisory

* Risk trend monitoring over time

* Hybrid AI chatbot for healthcare assistance

* Accepted conference paper (Phase II project)

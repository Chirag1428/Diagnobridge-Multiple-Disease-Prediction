
# main.py
import streamlit as st
import sys, os

# --- Page Config (MUST be first Streamlit command) ---
st.set_page_config(page_title="DiagnoBridge", page_icon="🩺", layout="centered")

# --- Make 'pages' folder importable ---
sys.path.append(os.path.join(os.path.dirname(__file__), "pages"))

from modules.login import show_login
from modules.app import run_app

# --- Ensure default state values ---
if "user" not in st.session_state:
    st.session_state["user"] = None
if "username" not in st.session_state:
    st.session_state["username"] = ""

# --- App flow ---
if st.session_state["user"] is None:
    show_login()
else:
    run_app()

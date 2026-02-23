import streamlit as st
from firebase_config import auth, db
import time
import json

def show_login():
    # --- Initialize session variables ---
    if "user" not in st.session_state:
        st.session_state["user"] = None
    if "username" not in st.session_state:
        st.session_state["username"] = ""

    # --- Handle saved query params (restore session on refresh) ---
    params = st.query_params
    if "user" in params and st.session_state["user"] is None:
        st.session_state["user"] = params["user"][0] if "user" in params else None
        st.session_state["username"] = params["username"][0] if "username" in params else "User"
        st.rerun()

    # Sidebar menu
    menu = ["Login", "Sign Up"]
    choice = st.sidebar.selectbox("Select Option", menu)

    # ---------------- SIGN UP ----------------
    if choice == "Sign Up":
        st.title("🩺 Create a New Account")
        username = st.text_input("Username")
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")

        if st.button("Create Account"):
            if not username or not email or not password:
                st.warning("Please fill all fields")
            else:
                try:
                    user = auth.create_user_with_email_and_password(email, password)
                    token = user['idToken']
                    db.child("users").child(user["localId"]).set(
                        {"username": username, "email": email},
                        token
                    )

                    st.success(f"Account created successfully! ✅ Welcome, {username}")
                    st.info("Go to Login menu to sign in.")
                except Exception as e:
                    error_str = str(e)
                    if "EMAIL_EXISTS" in error_str:
                        st.error("An account with this email already exists. Please go to Login instead.")
                    else:
                        st.error("Error creating account. Please try again.")
                        st.caption(error_str)  # optional for debugging


    # ---------------- LOGIN ----------------
    elif choice == "Login":
        st.title("🩺 Login to Your Account")
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")

        if st.button("Login") and st.session_state["user"] is None:
            if not email or not password:
                st.warning("Please fill all fields")
            else:
                try:
                    user = auth.sign_in_with_email_and_password(email, password)
                    token = user["idToken"]

                    username = (
                        db.child("users")
                        .child(user["localId"])
                        .child("username")
                        .get(token)
                        .val()
                    )

                    st.session_state["user"] = user["localId"]
                    st.session_state["username"] = username

                    st.query_params.update({"user": user["localId"], "username": username})

                    st.success(f"✅ Login successful! Welcome, {username}")
                    st.info("Redirecting to Health App... Please wait...")

                    with st.spinner("Loading your dashboard..."):
                        time.sleep(2)

                    st.rerun()

                except Exception as e:
                    # keep this simple for now
                    st.error("Login failed. Invalid login credentials. Please try again.")


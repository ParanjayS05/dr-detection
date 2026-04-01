import streamlit as st
import torch
import timm
import gdown
import os
from PIL import Image
import torchvision.transforms as transforms
import pyrebase
import datetime
import pandas as pd
import time
import google.generativeai as genai
from datetime import date
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet
import matplotlib.pyplot as plt

# -----------------------------
# UI CLEAN
# -----------------------------
st.markdown("""
<style>
a[href*="github.com"] {display:none;}
button[title="Edit this app"] {display:none;}
button[title="Star this repo"] {display:none;}
button[title="Share this app"] {display:none;}
footer {visibility:hidden;}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# GEMINI
# -----------------------------
genai.configure(api_key=st.secrets["AIzaSyA8DzYv0ume3BZW7-PEEG9at2Pi8bRUukg"])
model_gemini = genai.GenerativeModel("gemini-1.5-flash")
response = model_gemini.generate_content("What is diabetic retinopathy?")
st.write(response.text)

# -----------------------------
# FIREBASE
# -----------------------------
firebase_config = {
   # // For Firebase JS SDK v7.20.0 and later, measurementId is optional
  apiKey: "AIzaSyALcofpX68Ok6lA-TxOtmW0-sHx9WoMWeA",
  authDomain: "diabetese-detection-project.firebaseapp.com",
  projectId: "diabetese-detection-project",
  storageBucket: "diabetese-detection-project.firebasestorage.app",
  messagingSenderId: "264109211554",
  appId: "1:264109211554:web:5f46144b1b8695a0883cfb",
  measurementId: "G-GCR2Q4TY6D",
  databaseURL: "https://diabetese-detection-project-default-rtdb.firebaseio.com/"

}

firebase = pyrebase.initialize_app(firebase_config)
auth = firebase.auth()
db = firebase.database()

# -----------------------------
# ADMIN EMAIL
# -----------------------------
ADMIN_EMAILS = ["your_admin_email@gmail.com"]

# -----------------------------
# SESSION
# -----------------------------
if "user" not in st.session_state:
    st.session_state.user = None

# -----------------------------
# LOGIN
# -----------------------------
if not st.session_state.user:
    st.title("🔐 Login System")

    email = st.text_input("Email")
    password = st.text_input("Password", type="password")

    col1, col2 = st.columns(2)

    if col1.button("Login"):
        try:
            user = auth.sign_in_with_email_and_password(email, password)
            st.session_state.user = user
            st.rerun()
        except:
            st.error("Invalid credentials")

    if col2.button("Signup"):
        try:
            auth.create_user_with_email_and_password(email, password)
            st.success("Account created")
        except:
            st.error("Signup failed")

    st.stop()

# -----------------------------
# ROLE
# -----------------------------
user_email = st.session_state.user["email"]
role = "Admin" if user_email in ADMIN_EMAILS else "Patient"

st.sidebar.success(f"Logged in as {role}")

if st.sidebar.button("Logout"):
    st.session_state.user = None
    st.rerun()

# -----------------------------
# ADMIN PANEL
# -----------------------------
if role == "Admin":
    st.title("👨‍⚕️ Admin Dashboard")

    data = db.child("predictions").get()
    records = []

    if data.each():
        for item in data.each():
            records.append(item.val())

    if records:
        df = pd.DataFrame(records)

        st.subheader("📊 Prediction Distribution")
        st.bar_chart(df["prediction"].value_counts())

        st.subheader("👥 User Activity")
        st.bar_chart(df["email"].value_counts())

        st.subheader("🔁 Repeat Users")
        st.write(df["email"].value_counts()[df["email"].value_counts() > 1])

        # Graph Download
        plt.figure()
        df["prediction"].value_counts().plot(kind="bar")
        plt.savefig("graph.png")

        with open("graph.png", "rb") as f:
            st.download_button("📈 Download Graph", f)

# -----------------------------
# PATIENT PANEL
# -----------------------------
else:
    st.title("👁️ Diabetic Retinopathy Detection")

    # Patient Info
    st.subheader("👤 Patient Details")

    name = st.text_input("Full Name")
    dob = st.date_input("Date of Birth")
    blood_group = st.selectbox("Blood Group",
        ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"])

    def calculate_age(dob):
        today = date.today()
        years = today.year - dob.year
        days = (today - dob).days
        return years, days

    age_years, age_days = calculate_age(dob)
    st.write(f"Age: {age_years} years ({age_days} days)")

    # MODEL
    MODEL_PATH = "model.pth"
    MODEL_URL = "https://drive.google.com/uc?id=1yDdDELohhVrnI_SSRAQAbqkruoV0fBpw"

    if not os.path.exists(MODEL_PATH):
        gdown.download(MODEL_URL, MODEL_PATH)

    @st.cache_resource
    def load_model():
        model = timm.create_model('convnext_base', pretrained=False, num_classes=5)
        model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=False))
        model.eval()
        return model

    model = load_model()
    labels = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]

    file = st.file_uploader("Upload Retina Image")

    if file and name:
        img = Image.open(file).convert("RGB")
        st.image(img)

        image_path = "uploaded.jpg"
        img.save(image_path)

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

        input_tensor = transform(img).unsqueeze(0)

        with torch.no_grad():
            output = model(input_tensor)
            probs = torch.softmax(output, dim=1)
            pred = torch.argmax(probs, dim=1).item()

        st.success(labels[pred])

        # SAVE DB
        db.child("predictions").push({
            "email": user_email,
            "name": name,
            "prediction": labels[pred],
            "timestamp": str(datetime.datetime.now())
        })

        # PDF
        def generate_pdf():
            doc = SimpleDocTemplate("report.pdf")
            styles = getSampleStyleSheet()

            content = []
            content.append(Paragraph("Medical Report", styles["Title"]))
            content.append(Paragraph(f"Name: {name}", styles["Normal"]))
            content.append(Paragraph(f"DOB: {dob}", styles["Normal"]))
            content.append(Paragraph(f"Age: {age_years} years", styles["Normal"]))
            content.append(Paragraph(f"Blood Group: {blood_group}", styles["Normal"]))
            content.append(Paragraph(f"Prediction: {labels[pred]}", styles["Normal"]))
            content.append(Paragraph("Advice: Consult doctor", styles["Normal"]))
            content.append(RLImage(image_path, width=200, height=200))

            doc.build(content)
            return "report.pdf"

        pdf = generate_pdf()

        with open(pdf, "rb") as f:
            st.download_button("📄 Download Report", f)

    # -----------------------------
    # NETRA CHATBOT
    # -----------------------------
    st.sidebar.title("🤖 Netra")

    if "last_query_time" not in st.session_state:
        st.session_state.last_query_time = 0

    query = st.sidebar.text_input("Ask about eye health")

    def netra_ai(q):
        response = model_gemini.generate_content(q)
        return response.text

    if st.sidebar.button("Ask Netra"):
        if time.time() - st.session_state.last_query_time > 5:
            st.session_state.last_query_time = time.time()
            st.sidebar.write(netra_ai(query))
        else:
            st.sidebar.warning("Wait before next query")

    st.sidebar.warning("Consult doctor for medical advice")

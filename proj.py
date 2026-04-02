import streamlit as st
import torch
import timm
import gdown
import os
from PIL import Image
import torchvision.transforms as transforms
import google.generativeai as genai
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet
from datetime import date
import time

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="DR Detection", layout="centered")

# -----------------------------
# GEMINI API
# -----------------------------
genai.configure(api_key="YOUR_GEMINI_API_KEY")
model_gemini = genai.GenerativeModel("gemini-1.5-flash")

# -----------------------------
# TITLE
# -----------------------------
st.title("👁️ Diabetic Retinopathy Detection System")

# -----------------------------
# ABOUT SECTION
# -----------------------------
st.sidebar.title("🧠 About")

st.sidebar.write("""
Diabetic Retinopathy is a serious eye condition caused by diabetes.

It damages the retina and can lead to:
- Blurred vision
- Vision loss
- Blindness if untreated

Early detection is very important.
""")

# -----------------------------
# PATIENT DETAILS
# -----------------------------
st.subheader("👤 Patient Details")

name = st.text_input("Full Name")
dob = st.date_input("Date of Birth")

blood_group = st.selectbox(
    "Blood Group",
    ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"]
)

# -----------------------------
# AGE CALCULATION
# -----------------------------
def calculate_age(dob):
    today = date.today()
    years = today.year - dob.year
    days = (today - dob).days
    return years, days

age_years, age_days = calculate_age(dob)
st.write(f"Age: {age_years} years ({age_days} days)")

# -----------------------------
# MODEL LOAD
# -----------------------------
MODEL_PATH = "model.pth"
MODEL_URL = "https://drive.google.com/uc?id=1yDdDELohhVrnI_SSRAQAbqkruoV0fBpw"

if not os.path.exists(MODEL_PATH):
    st.info("Downloading model...")
    gdown.download(MODEL_URL, MODEL_PATH)
    st.success("Model ready")

@st.cache_resource
def load_model():
    model = timm.create_model('convnext_base', pretrained=False, num_classes=5)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=False))
    model.eval()
    return model

model = load_model()

labels = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]

# -----------------------------
# IMAGE UPLOAD
# -----------------------------
file = st.file_uploader("Upload Retina Image", type=["jpg", "png", "jpeg"])

if file and name:
    img = Image.open(file).convert("RGB")
    st.image(img, caption="Uploaded Image")

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

    st.success(f"Prediction: {labels[pred]}")

    # -----------------------------
    # PDF GENERATION
    # -----------------------------
    def generate_pdf():
        doc = SimpleDocTemplate("report.pdf")
        styles = getSampleStyleSheet()

        content = []

        content.append(Paragraph("Diabetic Retinopathy Report", styles["Title"]))
        content.append(Spacer(1, 10))

        content.append(Paragraph(f"Name: {name}", styles["Normal"]))
        content.append(Paragraph(f"DOB: {dob}", styles["Normal"]))
        content.append(Paragraph(f"Age: {age_years} years", styles["Normal"]))
        content.append(Paragraph(f"Blood Group: {blood_group}", styles["Normal"]))
        content.append(Spacer(1, 10))

        content.append(Paragraph(f"Prediction: {labels[pred]}", styles["Heading2"]))
        content.append(Paragraph("Advice: Consult an ophthalmologist.", styles["Normal"]))

        content.append(Spacer(1, 10))
        content.append(RLImage(image_path, width=200, height=200))

        doc.build(content)
        return "report.pdf"

    pdf = generate_pdf()

    with open(pdf, "rb") as f:
        st.download_button("📄 Download Report", f)

# -----------------------------
# NETRA CHATBOT
# -----------------------------
st.sidebar.title("🤖 Netra AI")

if "last_query_time" not in st.session_state:
    st.session_state.last_query_time = 0

query = st.sidebar.text_input("Ask about eye health")

def netra_ai(q):
    prompt = f"You are Netra, an eye specialist AI. Answer clearly: {q}"
    response = model_gemini.generate_content(prompt)
    return response.text

if st.sidebar.button("Ask Netra"):
    if time.time() - st.session_state.last_query_time > 5:
        st.session_state.last_query_time = time.time()
        st.sidebar.write(netra_ai(query))
    else:
        st.sidebar.warning("Wait a few seconds")

st.sidebar.warning("⚠️ This AI is for guidance only. Consult a doctor.")

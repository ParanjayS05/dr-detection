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
# CUSTOM CHAT UI CSS
# -----------------------------
st.markdown("""
<style>
.chat-container {
    display: flex;
    align-items: flex-start;
    margin-bottom: 10px;
}
.chat-avatar {
    width: 40px;
    height: 40px;
    border-radius: 50%;
    background-color: #4CAF50;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 20px;
    margin-right: 10px;
}
.chat-bubble {
    background-color: #262730;
    padding: 10px 15px;
    border-radius: 15px;
    max-width: 80%;
}
.user-bubble {
    background-color: #4CAF50;
    color: white;
    margin-left: auto;
}
</style>
""", unsafe_allow_html=True)

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
# ABOUT
# -----------------------------
st.sidebar.title("🧠 About")
st.sidebar.write("""
Diabetic Retinopathy damages the retina due to diabetes.

Effects:
- Blurred vision
- Floaters
- Vision loss
- Blindness (if untreated)

Early detection is critical.
""")

# -----------------------------
# PATIENT DETAILS
# -----------------------------
st.subheader("👤 Patient Details")

name = st.text_input("Full Name")

dob = st.date_input(
    "Date of Birth",
    min_value=date(1900, 1, 1),
    max_value=date.today()
)

blood_group = st.selectbox(
    "Blood Group",
    ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"]
)

# -----------------------------
# AGE
# -----------------------------
def calculate_age(dob):
    today = date.today()
    years = today.year - dob.year
    days = (today - dob).days
    return years, days

age_years, age_days = calculate_age(dob)
st.write(f"Age: {age_years} years ({age_days} days)")

# -----------------------------
# MODEL
# -----------------------------
MODEL_PATH = "model.pth"
MODEL_URL = "https://drive.google.com/uc?id=1yDdDELohhVrnI_SSRAQAbqkruoV0fBpw"

if not os.path.exists(MODEL_PATH):
    st.info("Downloading model...")
    gdown.download(MODEL_URL, MODEL_PATH)

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

    st.success(f"Prediction: {labels[pred]}")

    # -----------------------------
    # PDF
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
        content.append(Paragraph("Advice: Consult doctor", styles["Normal"]))

        content.append(Spacer(1, 10))
        content.append(RLImage(image_path, width=200, height=200))

        doc.build(content)
        return "report.pdf"

    pdf = generate_pdf()

    with open(pdf, "rb") as f:
        st.download_button("📄 Download Report", f, file_name="DR_Report.pdf")

# -----------------------------
# CHATBOT
# -----------------------------
st.subheader("💬 Chat with Netra")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_query = st.text_input("Ask about eye health")

def netra_ai(q):
    prompt = f"You are Netra, an eye specialist AI. Answer clearly: {q}"
    response = model_gemini.generate_content(prompt)
    return response.text

if "last_query_time" not in st.session_state:
    st.session_state.last_query_time = 0

if st.button("Ask Netra"):
    if user_query:
        if time.time() - st.session_state.last_query_time > 5:
            st.session_state.last_query_time = time.time()

            st.session_state.chat_history.append(("user", user_query))
            reply = netra_ai(user_query)
            st.session_state.chat_history.append(("bot", reply))
        else:
            st.warning("Wait a few seconds")

# -----------------------------
# DISPLAY CHAT
# -----------------------------
for role, message in st.session_state.chat_history:
    if role == "user":
        st.markdown(f"""
        <div class="chat-container" style="justify-content:flex-end;">
            <div class="chat-bubble user-bubble">{message}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="chat-container">
            <div class="chat-avatar">👁️</div>
            <div class="chat-bubble">{message}</div>
        </div>
        """, unsafe_allow_html=True)

st.warning("⚠️ AI is for guidance only. Consult doctor.")

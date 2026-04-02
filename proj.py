import streamlit as st
import torch
import timm
import gdown
import os
from PIL import Image
import torchvision.transforms as transforms
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet
from datetime import date

# -----------------------------
# PAGE CONFIG
# -----------------------------

st.set_page_config(page_title="DR Detection", layout="centered")

st.markdown("""
<style> header {visibility: hidden;} 
a[href*="github"] {display: none !important;} 
button[title*="Fork"] {display: none !important;} 
</style>
""",unsafe_allow_html=True)

# -----------------------------
# TITLE
# -----------------------------
st.title(" Diabetic Retinopathy Detection System")

# -----------------------------
# ABOUT SECTION
# -----------------------------
st.sidebar.title("About")

st.sidebar.write("""
Diabetic retinopathy is a medical condition where chronic high blood sugar levels damage the delicate blood vessels in the retina, the light-sensitive tissue at the back of the eye. 
Over time, these vessels can swell, leak fluid, or close off entirely, sometimes triggering the growth of abnormal new vessels that further interfere with vision.

Effects:
1.Blurred Vision (The Macula)

 Where it happens: At the very center of your retina, in a tiny area called the macula.

 How it happens: High blood sugar causes blood vessels to leak fluid or protein into the macula, making it swell (a condition called Macular Edema). Since the macula is responsible for sharp, straight-ahead vision, the world starts to look like you're looking through a dense fog or a smudged window.

2. Floaters (The Vitreous)
Where it happens: In the vitreous humor, which is the clear, jelly-like substance filling the middle of your eye.

How it happens: In advanced stages, fragile new blood vessels grow and then break, leaking blood into this "jelly." This creates dark spots, cobweb-like strings, or red streaks that "float" across your field of vision.

3. Blindness (Retinal Damage & Detachment)
Where it happens: Across the entire retinal surface and the optic nerve.

How it happens: * Total Blockage: Severe bleeding can completely fill the eye, blocking all light from reaching the retina.

Retinal Detachment: Scar tissue from the damaged vessels can pull the retina away from the back of the eye, causing permanent vision loss if not treated immediately.

Glaucoma: New vessels can block fluid drainage, causing high pressure that destroys the optic nerve (the "cable" that sends images to your brain).

Early detection is very important.
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
# ADVICE FUNCTION (NEW 🔥)
# -----------------------------
def get_advice(prediction):
    if prediction == "No DR":
        return "No signs of diabetic retinopathy detected. Maintain healthy lifestyle and regular eye checkups."
    elif prediction == "Mild":
        return "Mild diabetic retinopathy detected. Monitor regularly and consult a doctor if symptoms increase."
    elif prediction == "Moderate":
        return "Moderate diabetic retinopathy detected. It is recommended to consult an ophthalmologist."
    elif prediction == "Severe":
        return "Severe diabetic retinopathy detected. Immediate medical attention is required."
    else:
        return "Proliferative diabetic retinopathy detected. Urgent treatment is necessary to prevent vision loss."

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

    prediction_label = labels[pred]
    advice = get_advice(prediction_label)

    # -----------------------------
    # OUTPUT
    # -----------------------------
    st.success(f"Prediction: {prediction_label}")
    st.info(f"Advice: {advice}")

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

        content.append(Paragraph(f"Prediction: {prediction_label}", styles["Heading2"]))
        content.append(Paragraph(f"Advice: {advice}", styles["Normal"]))

        content.append(Spacer(1, 10))
        content.append(RLImage(image_path, width=200, height=200))

        doc.build(content)
        return "report.pdf"

    pdf = generate_pdf()

    with open(pdf, "rb") as f:
        st.download_button(
            "📄 Download Report",
            f,
            file_name="DR_Report.pdf"
        )

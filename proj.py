import streamlit as st
import torch
import timm
import gdown
import os
from PIL import Image
import torchvision.transforms as transforms

# -----------------------------
# CONFIG
# -----------------------------
MODEL_PATH = "model.pth"
MODEL_URL = "https://drive.google.com/uc?id=1yDdDELohhVrnI_SSRAQAbqkruoV0fBpw"

labels = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]

# -----------------------------
# Download model (FIXED)
# -----------------------------
if not os.path.exists(MODEL_PATH):
    st.info("Downloading model... (first time only ⏳)")
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
    st.success("Model downloaded ✅")

# -----------------------------
# Load model (FIXED for PyTorch 2.6+)
# -----------------------------
@st.cache_resource
def load_model():
    model = timm.create_model('convnext_base', pretrained=False, num_classes=5)
    
    # 🔥 Important fix
    state_dict = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    model.load_state_dict(state_dict)
    
    model.eval()
    return model

model = load_model()

# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="DR Detection", layout="centered")

st.title("👁️ Diabetic Retinopathy Detection")

st.sidebar.title("About")
st.sidebar.write("Upload a retinal image to detect Diabetic Retinopathy severity using AI")

file = st.file_uploader("Upload Retina Image", type=["jpg", "png", "jpeg"])

if file:
    try:
        img = Image.open(file).convert("RGB")
        st.image(img, caption="Uploaded Image", use_column_width=True)

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

        input_tensor = transform(img).unsqueeze(0)

        with torch.no_grad():
            output = model(input_tensor)
            probs = torch.softmax(output, dim=1)
            pred = torch.argmax(probs, dim=1).item()

        # Prediction
        st.success(f"Prediction: {labels[pred]}")

        # Confidence scores
        st.subheader("Confidence Scores")
        for i, p in enumerate(probs[0]):
            st.write(f"{labels[i]}: {p.item()*100:.2f}%")

    except Exception as e:
        st.error("Error processing image")
        st.text(str(e))

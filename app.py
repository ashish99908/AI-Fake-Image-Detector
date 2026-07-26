import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import tempfile
from PIL import Image
from tensorflow.keras.models import load_model


# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="AI Fake Image Detector",
    page_icon="🖼️",
    layout="wide"
)

# -----------------------------
# Load CSS
# -----------------------------
try:
    with open("style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except:
    pass

# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_ai_model():
    return load_model("fake_real_classifier.keras")

model = load_ai_model()

IMG_SIZE = (224,224)

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.image(
    "https://img.icons8.com/color/96/artificial-intelligence.png",
    width=80
)

st.sidebar.title("AI Fake Detector")

page = st.sidebar.radio(
    "Select Option",
    [
        "Single Image",
        "Multiple Images",
        "About"
    ]
)

st.sidebar.markdown("---")

st.sidebar.info(
"""
CNN Model

TensorFlow + Keras

Binary Classification

Classes

✔ REAL

✔ FAKE
"""
)

# -----------------------------
# Title
# -----------------------------
st.markdown("""
# 🧠 AI Fake Image Detector

### Detect AI Generated Images using Deep Learning

---
""")
# ============================
# Prediction Function
# ============================

def predict_image(image):

    image = image.resize(IMG_SIZE)

    img = np.array(image)

    img = img / 255.0

    img = np.expand_dims(img, axis=0)

    prediction = float(model.predict(img, verbose=0)[0][0])


    if prediction > 0.5:
        label = "REAL"
        confidence = prediction
        
    else:
        label = "FAKE"
        confidence = 1 - prediction



    return label, confidence * 100


# ============================
# Single Image Page
# ============================

# ============================
# Single Image Page
# ============================

if page == "Single Image":

    st.header("📷 Upload an Image")

    uploaded_file = st.file_uploader(
        "Choose an Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:


        from PIL import ImageOps

        image = ImageOps.exif_transpose(Image.open(uploaded_file)).convert("RGB")

        col1, col2 = st.columns(2)

        with col1:

            st.image(
                image,
                caption="Uploaded Image",
                use_container_width=True
            )

        with st.spinner("🤖 AI is analyzing the image..."):

            label, confidence = predict_image(image)
            

        with col2:

            if label == "REAL":

                st.markdown("""
<div style="
background:#1e4620;
padding:25px;
border-radius:15px;
text-align:center;
">

<h2 style="color:white;">✅ REAL IMAGE</h2>

</div>
""", unsafe_allow_html=True)

            else:

                st.markdown("""
<div style="
background:#6b1d1d;
padding:25px;
border-radius:15px;
text-align:center;
">

<h2 style="color:white;">❌ AI GENERATED</h2>

</div>
""", unsafe_allow_html=True)

            st.metric(
                "Confidence",
                f"{confidence:.2f}%"
            )

            st.markdown("### Prediction Probability")

            progress = float(confidence) / 100
            st.metric(
                label="🎯 Confidence",
                value=f"{confidence:.2f}%"
                )
            st.progress(float(confidence) / 100)


            if confidence > 95:

               st.success("Very High Confidence")
            elif confidence > 80:
               st.info("High Confidence")
            else:
               st.warning("Low Confidence")

               st.markdown("---")

            if label == "REAL":

                st.info("This image appears to be an authentic photograph.")

            else:

                st.warning("This image appears to be AI Generated.")
                # ==========================================
# Multiple Image Detection
# ==========================================

# ==========================================
# Multiple Image Detection
# ==========================================

if page == "Multiple Images":

    st.header("📂 Batch Image Detection")

    uploaded_files = st.file_uploader(
        "Upload Multiple Images",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )

    if uploaded_files:

        results = []
        fake = 0
        real = 0

        progress = st.progress(0.0)

        for i, file in enumerate(uploaded_files):

            image = Image.open(file).convert("RGB")

            label, confidence = predict_image(image)

            # Gallery
            with st.expander(f"🖼️ {file.name}"):

                st.image(image, use_container_width=True)

                if label == "REAL":
                    st.success(f"✅ REAL ({confidence:.2f}%)")
                else:
                    st.error(f"❌ AI GENERATED ({confidence:.2f}%)")

            # Count
            if label == "REAL":
                real += 1
            else:
                fake += 1

            # Save Result
            results.append({
                "Image": file.name,
                "Prediction": label,
                "Confidence": f"{confidence:.2f}%"
            })

            progress.progress((i + 1) / len(uploaded_files))

        st.success("✅ Batch Analysis Complete")

        # Statistics
        col1, col2, col3 = st.columns(3)

        col1.metric("📁 Total Images", len(uploaded_files))
        col2.metric("✅ Real Images", real)
        col3.metric("❌ Fake Images", fake)

        st.markdown("---")

        # Pie Chart
        fig, ax = plt.subplots(figsize=(5, 5))

        ax.pie(
            [real, fake],
            labels=["Real", "Fake"],
            autopct="%1.1f%%",
            startangle=90,
            colors=["#2ecc71", "#e74c3c"]
        )

        ax.axis("equal")

        st.pyplot(fig)

        st.markdown("---")

        # Prediction Table
        df = pd.DataFrame(results)
        os.makedirs("reports", exist_ok=True)

        df.to_csv("reports/history.csv", index=False)

        st.dataframe(df, use_container_width=True)

        # Download CSV
        csv = df.to_csv(index=False).encode("utf-8")

        st.download_button(
            "📄 Download CSV Report",
            csv,
            "prediction_report.csv",
            "text/csv"
        )
# ==========================================
# About Page
# ==========================================

if page == "About":

    st.header("ℹ️ About")

    st.markdown("""
# AI Fake Image Detector

This project detects whether an uploaded image is **REAL** or **AI GENERATED**
using a Convolutional Neural Network (CNN).

---

## Features

- ✅ Single Image Detection
- ✅ Multiple Image Detection
- ✅ Batch Processing
- ✅ Confidence Score
- ✅ Pie Chart Analytics
- ✅ CSV Report Download

---

## Model

- TensorFlow
- Keras
- CNN
- Binary Classification

---

## Classes

- REAL
- FAKE

---

## Developed By

BCA Deep Learning Project
""")
st.markdown("---")

st.markdown(
"""
<div style='text-align:center;color:gray;'>

Developed using ❤️ TensorFlow + Streamlit

</div>
""",
unsafe_allow_html=True
)

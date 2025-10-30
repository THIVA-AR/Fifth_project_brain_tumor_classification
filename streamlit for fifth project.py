import streamlit as st
import base64
 
st.set_page_config(layout="wide")
 
def get_base64_image(image_path):
     with open(image_path, "rb") as f:
         data = f.read()
     return base64.b64encode(data).decode()
 
image_path = r"C:\Users\user\Documents\guvi\guvi project 1\fifth project\bg for fifth project.png"
encoded = get_base64_image(image_path)
 
 # Inject CSS to add the background
st.markdown(
     f"""
     <style>
     .stApp {{
         background-image: url("data:image/webp;base64,{encoded}");
         background-size: cover;
         background-repeat: no-repeat;
         background-attachment: fixed;
         background-position: center;
     }}
     </style>
     """,
     unsafe_allow_html=True
)
st.markdown("""
    <div style="
        background-color: black;
        border-radius: 16px;
        padding: 18px 8px;
        margin-bottom: 24px;
        text-align: center;
        display: inline-block;
        width: 100%;
    ">
        <h1 style="
            color: blue;
            -webkit-text-stroke: 2px pink;
            font-weight: bold;
            margin: 0;
        ">
            Brain Tumor MRI Classification
        </h1>
    </div>
""", unsafe_allow_html=True)

import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import json


# Load class names saved during training
with open(r"C:\Users\user\Documents\guvi\guvi project 1\fifth project\class_names.json") as f:
    class_names = json.load(f)

# Load the trained model
model = load_model(r"C:\Users\user\Documents\guvi\guvi project 1\fifth project\transfer_resnet_brain_mri.keras")

uploaded_file = st.file_uploader("Upload a brain MRI image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load image and resize
    img = load_img(uploaded_file, target_size=(224, 224))
    st.image(img, caption='Uploaded MRI Image', use_container_width=True)

    # Preprocess for model
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    preds = model.predict(img_array)
    pred_index = np.argmax(preds[0])
    pred_class = class_names[pred_index]
    confidence = preds[0][pred_index]

    # Display prediction and confidence
    st.write(f"### Predicted Tumor Type: {pred_class}")
    st.write(f"Confidence: {confidence:.2%}")

    # Optional: Show all class probabilities
    st.write("#### Confidence Scores for All Classes:")
    for i, cname in enumerate(class_names):
        st.write(f"{cname}: {preds[0][i]:.2%}")

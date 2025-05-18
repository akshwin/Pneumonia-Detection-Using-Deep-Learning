import streamlit as st
from PIL import Image
from tensorflow.keras.utils import load_img, img_to_array
import numpy as np
from keras.models import load_model
import os

# Load the pre-trained model
model = load_model("pneumonia.h5")

# Labels for prediction output
labels = {0: 'No Pneumonia', 1: 'Pneumonia'}
pneumonia = {'Pneumonia'}

# Image Preprocessing and Prediction
def processed_img(img_path):
    img = load_img(img_path, target_size=(224, 224, 3))
    img = img_to_array(img)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction)
    confidence = round(np.max(prediction) * 100, 2)
    label = labels[predicted_class]
    return label.capitalize(), confidence

# Main App Function
def run():
    st.set_page_config(page_title="Pneumonia Detection", layout="centered")
    
    # Title and Description
    st.markdown(
        "<h1 style='text-align: center; color: #FF5722;'>Pneumonia Prediction from X-Ray</h1>", 
        unsafe_allow_html=True
    )
    st.markdown(
        "<p style='text-align: center;'>Upload a chest X-Ray image and let the AI detect if there is pneumonia.</p>", 
        unsafe_allow_html=True
    )
    
    # Sidebar
    with st.sidebar:
        
        st.markdown("## 🩺 Project Info")
        st.markdown("""This application uses deep learning to detect pneumonia from chest X-ray images.  
        Powered by **Convolutional Neural Networks (CNNs)** and trained on medical data.""")

        with st.expander("🔍 Model Details"):
            st.markdown("""
            - **Architecture Used**:  
              - Convolutional Neural Networks (CNNs)  
            - **Best Accuracy**: 94.5%
            """)

        with st.expander("📂 Classes Detected"):
            st.markdown("""
            - 🩸 **Pneumonia**  
            - 🏥 **No Pneumonia**
            """)

        with st.expander("📁 Dataset Info"):
            st.markdown("""
            - The model is trained on chest X-ray images from public datasets.  
            - Preprocessed and resized to 224x224 pixels for prediction.
            """)

        st.markdown("---")
        st.markdown("👨‍💻 Developed by Akshwin T")
        st.markdown("📬 Contact: [akshwint.2003@gmail.com](mailto:akshwint.2003@gmail.com)")

    # File Upload
    img_file = st.file_uploader("📤 Upload a Chest X-Ray Image", type=['jpg', 'jpeg', 'png'])

    if img_file is not None:
        upload_dir = "./upload_image"
        os.makedirs(upload_dir, exist_ok=True)
        save_path = os.path.join(upload_dir, img_file.name)

        # Save the uploaded file
        with open(save_path, "wb") as f:
            f.write(img_file.getbuffer())

        # Display the uploaded image
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(Image.open(save_path), caption='🖼 Uploaded X-Ray Image', width=300)

        # Make Prediction
        with st.spinner("🩺 Analyzing the X-ray image..."):
            result, confidence = processed_img(save_path)

        st.markdown("### 🔍 Prediction Result")
        if result in pneumonia:
            st.error(f"🚨 **PNEUMONIA DETECTED!**")
        else:
            st.success("✅ **No Pneumonia Detected**")
            st.balloons()

        st.info(f"📊 **Confidence: {confidence}%**")

        # Clean up the uploaded image after processing
        os.remove(save_path)

# Run the app
if __name__ == "__main__":
    run()

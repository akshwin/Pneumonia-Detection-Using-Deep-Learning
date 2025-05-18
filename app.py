<<<<<<< HEAD
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
=======
import streamlit as st 
from PIL import Image
from tensorflow.keras.utils import load_img,img_to_array
import numpy as np 
from keras.models import load_model 


model = load_model ("pneumonia.h5")
labels ={0:'No Pneumonia',1:'Pneumonia'}
pneumonia = {'Pneumonia'}

def processed_img(img_path):
    img=load_img(img_path,target_size=(224,224,3))
    img=img_to_array(img)
    img=img/255
    img=np.expand_dims(img,[0])
    answer=model.predict(img)
    y_class = answer.argmax(axis=-1)
    y = " ".join(str(x) for x in y_class)
    y = int(y)
    res = labels[y]
    return res.capitalize()

def run():
    st.title("Pneumonia Detector from X-Ray🫁")
    st.subheader("Upload the MRI Image:")
    
    st.sidebar.header("About the projet :")
    st.sidebar.write("📌 The model identifies whether X-Ray uploded contains Pneumonia or not. ")
    st.sidebar.write("📌 The project is developed using a Deep learning algorithm named CNN .")
    st.sidebar.write("📌 The model acheived an accuracy of 88 percent")
    
    img_file = st.file_uploader("Choose an image",type=['jpg','jpeg','png'])

    if img_file is not None :
        img  = Image.open(img_file).resize((250,250))
        st.image(img)
        save_image_path = './upload_image/'+img_file.name
        with open(save_image_path,"wb") as f:
            f.write(img_file.getbuffer())

        if img_file is not None :
            result = processed_img(save_image_path)
            if result in pneumonia :
                st.error('**PNEUMONIA DETECTED!!**')
            else :
                st.success('**NO PNEUMONIA!!**')
                st.balloons()
run()
>>>>>>> 6f4b461c0b981195d80c439ff6b5487c5659ebee

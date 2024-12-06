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
    st.sidebar.write("📌 The model acheived an accuracy of 88%.")
    
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

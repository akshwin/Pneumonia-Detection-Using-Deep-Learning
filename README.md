# Pneumonia Detection from X-Ray Images

This Streamlit application uses a deep learning model to predict pneumonia from X-ray images. The model, "pneumonia.h5," is employed for analysis. The user-friendly interface allows users to upload X-ray images and receive instant predictions.

## Overview

- **Model**: Utilizes a pre-trained deep learning model for pneumonia prediction.
- **Labels**: The model classifies images into "No Pneumonia" (0) and "Pneumonia" (1).
- **User Interface**: Created using Streamlit for easy interaction and quick analysis.

## Setup and Execution

### Prerequisites

- Ensure Python is installed on your system.
- Install required packages using the following command:

    ```bash
    pip install streamlit Pillow numpy tensorflow
    ```

### Execution

1. **Clone the Repository:**

    ```bash
    git clone https://github.com/your_username/pneumonia-detection.git
    cd pneumonia-detection
    ```

2. **Run the Application:**

    ```bash
    streamlit run app.py
    ```

3. **Upload an Image:**
    - Use the file uploader to select an X-ray image (JPG, JPEG, or PNG).

4. **View Prediction:**
    - The processed image and prediction result will be displayed.
    - An error message indicates "Pneumonia Detected," and success triggers celebratory balloons for "No Pneumonia."
  
   
5. **Run the Application Online :**

https://pneumonia-predictor-app.streamlit.app/


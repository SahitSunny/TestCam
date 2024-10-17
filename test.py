import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
import time

# Load the model
model = load_model("facemodel.h5")

# Function to predict acne
def detect_acne(frame):
    face = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face = cv2.resize(face, (224, 224))
    face = img_to_array(face)
    face = preprocess_input(face)
    face = np.expand_dims(face, axis=0)
    (acne, withoutAcne) = model.predict(face)[0]
    return acne, withoutAcne

# Streamlit app layout
st.title("Acne Detection App")
st.write("This app uses a webcam to detect acne in real time.")

# Start video capture
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    st.error("Error: Could not open webcam.")
else:
    st.success("Webcam is open. Click below to start detection.")

# Button to start detection
if st.button("Start Detection"):
    last_prediction_time = time.time()
    acne_detected = False

    # Run detection loop
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        if not ret:
            st.error("Error: Could not read frame.")
            break

        current_time = time.time()

        # Check if 5 seconds have passed
        if current_time - last_prediction_time >= 5:
            acne, withoutAcne = detect_acne(frame)
            label = "Acne" if acne > withoutAcne else "No Acne"
            confidence = max(acne, withoutAcne) * 100

            # Display the prediction result
            st.write(f"Prediction: {label}, Confidence: {confidence:.2f}%")

            acne_detected = label == "Acne"
            last_prediction_time = current_time

        # Display the frame
        st.image(frame, channels="RGB", use_column_width=True)

        # Break the loop if 'q' is pressed (you can replace this with a Stop button)
        if st.button("Stop Detection"):
            break

    # Release the capture
    cap.release()

st.write("Press 'q' to exit or click 'Stop Detection'.")

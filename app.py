import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
import os
if not os.path.exists("face_mask_model.h5"):
    st.error("Model file not found! Please ensure 'face_mask_model.h5' is in the root directory.")
    st.stop()
# Load the trained model
model = tf.keras.models.load_model("face_mask_model.h5")

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Streamlit App Title
st.title("😷 Real-Time Face Mask Detection")

# Checkbox to start/stop webcam
run = st.checkbox('Start Camera')

# Display webcam frames in Streamlit
FRAME_WINDOW = st.image([])

# Initialize webcam
cap = cv2.VideoCapture(0)

# Main loop
while run:
    ret, frame = cap.read()
    if not ret:
        st.warning("Could not access webcam.")
        break

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces
    faces = face_cascade.detectMultiScale(frame_rgb, 1.1, 4)

    for (x, y, w, h) in faces:
        face = frame_rgb[y:y+h, x:x+w]
        face_resized = cv2.resize(face, (150, 150))
        face_array = np.expand_dims(face_resized, axis=0) / 255.0

        # Prediction
        prediction = model.predict(face_array)[0][0]

        # Label and color based on prediction
        label = "Mask Detected" if prediction < 0.5 else "No Mask Detected"
        color = (0, 255, 0) if prediction < 0.5 else (255, 0, 0)

        # Draw rectangle and label on frame
        cv2.rectangle(frame_rgb, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame_rgb, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # Update frame in Streamlit
    FRAME_WINDOW.image(frame_rgb, channels="RGB")

# Release webcam on exit
cap.release()

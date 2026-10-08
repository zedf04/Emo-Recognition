import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
st.set_page_config(
    page_title="Facial Emotion Recognition",
    page_icon="😊",
    layout="centered"
)
st.title("😊 Facial Emotion Recognition")
st.write(
    "Upload a facial image and the ResNet50 model will detect the "
    "most likely emotion."
)
model = tf.keras.models.load_model("/mnt/c/Users/sagni/EMO-Recognition/Models/resnet50.keras")
face_detector = cv2.FaceDetectorYN.create(
    "models/face_detection_yunet_2023mar.onnx",
    "",
    (320, 320)
)
class_names = [
    'angry',
    'disgust',
    'fear',
    'happy',
    'neutral',
    'sad',
    'surprise'
]
st.title("Facial Emotion Recognition")
uploaded_file = st.file_uploader(
    "Upload a facial image",
    type=["jpg", "jpeg", "png"]
)
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image_array = np.array(image.convert("RGB"))
    height, width = image_array.shape[:2]
    face_detector.setInputSize((width, height))
    _, faces = face_detector.detect(image_array)
    st.write("Faces detected:", 0 if faces is None else len(faces))
    if faces is not None:
        x, y, w, h = faces[0][:4].astype(int)
        display_image = image_array.copy()
        cv2.rectangle(
            display_image,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )
        face = image_array[y:y + h, x:x + w]
        st.image(
            display_image,
            caption="Detected Face",
            channels="RGB"
        )
        face = cv2.cvtColor(face, cv2.COLOR_RGB2GRAY)
        face = cv2.resize(face, (48, 48))
        face = np.stack([face] * 3, axis=-1)
        face = np.expand_dims(face, axis=0)
        predictions = model.predict(face, verbose=0)
        predicted_index = np.argmax(predictions[0])
        predicted_emotion = class_names[predicted_index]
        confidence = predictions[0][predicted_index]
        st.success(
            f"Predicted Emotion: {predicted_emotion.capitalize()}  "
            f"({confidence:.2%} confidence)"
        )
        st.subheader("Emotion Probabilities")
        probabilities = predictions[0]
        for emotion, probability in zip(class_names, probabilities):
            st.write(f"{emotion.capitalize()}: {probability:.2%}")
            st.progress(float(probability))

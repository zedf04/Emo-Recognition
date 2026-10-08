import cv2
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model("models/resnet50.keras")

class_names = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise"
]

face_detector = cv2.FaceDetectorYN.create(
    "models/face_detection_yunet_2023mar.onnx",
    "",
    (320, 320)
)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to capture frame")
        break

    height, width = frame.shape[:2]

    face_detector.setInputSize((width, height))

    _, faces = face_detector.detect(frame)

    if faces is not None:
        for face in faces:

            x, y, w, h = face[:4].astype(int)

            # Crop face
            face_img = frame[y:y+h, x:x+w]

            if face_img.size == 0:
                continue

            # Convert BGR → grayscale
            face_img = cv2.cvtColor(
                face_img,
                cv2.COLOR_BGR2GRAY
            )

            # Resize to model input
            face_img = cv2.resize(
                face_img,
                (48, 48)
            )

            # Grayscale → 3 channels
            face_img = np.stack(
                [face_img] * 3,
                axis=-1
            )

            # Add batch dimension
            face_img = np.expand_dims(
                face_img,
                axis=0
            )

            # Predict emotion
            predictions = model.predict(
                face_img,
                verbose=0
            )
            print(
                dict(
                    zip(
                        class_names,
                        predictions[0].round(3)
                    )
                )
            )
            predicted_index = np.argmax(
                predictions[0]
            )

            emotion = class_names[predicted_index]

            confidence = predictions[0][predicted_index]

            # Draw face box
            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

            # Draw prediction
            label = f"{emotion.capitalize()} {confidence:.0%}"

            cv2.putText(
                frame,
                label,
                (x, max(30, y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

    cv2.imshow("Facial Emotion Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
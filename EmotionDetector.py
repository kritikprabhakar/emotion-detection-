import cv2
import numpy as np
from deepface import DeepFace


# Load the pre-trained emotion detection model
emotion_model = DeepFace.build_model('Emotion')

# Load the face cascade classifier
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise Exception("Could not open video device")

while True:
    ret, frame = cap.read()

    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        face_roi = cv2.resize(roi_gray, (48, 48))  # Resize the face ROI for emotion detection
        face_roi = np.expand_dims(face_roi, axis=0) / 255.0  # Normalize

        Predictions = emotion_model.predict(face_roi)
        emotion_label = np.argmax(Predictions)
        emotion = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"][emotion_label]

        x1, y1, w1, h1 = x, y - 35, 175, 75
        cv2.rectangle(frame, (x1, y1), (x1 + w1, y1 + h1), (0, 0, 0), -1)
        cv2.putText(frame, emotion, (x1 + int(w1/10), y1 + int(h1/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255))

    cv2.imshow('Camera Feed', frame)

    # Press 'q' to exit the loop and close the window
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    

cap.release()
cv2.destroyAllWindows()


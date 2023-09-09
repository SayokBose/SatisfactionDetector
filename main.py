import cv2
import random
import pandas as pd
import numpy as np  # Add this import for numpy
from deepface import DeepFace
from datetime import datetime
import face_recognition as fr

class Face:
    def __init__(self, encoder):
        self.encoder = encoder
        self.emotions = []

    def add_emotion(self, emotion):
        self.emotions.append(emotion)

    def get_mode_emotion(self):
        if self.emotions:
            return mode(self.emotions)
        else:
            return "No emotions detected"


# Load the pre-trained emotion detection model
model = DeepFace.build_model("Emotion")

# Define emotion labels
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

# Load the pre-trained deep learning-based face detection model from OpenCV
face_detector = cv2.dnn.readNetFromCaffe("deploy.prototxt.txt", "res10_300x300_ssd_iter_140000.caffemodel")

# Start capturing video
cap = cv2.VideoCapture(0)

# Create an empty DataFrame to store satisfaction data
satisfaction_data = pd.DataFrame(columns=['Timestamp', 'Satisfaction'])

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Resize the frame for faster processing (optional)
    frame = cv2.resize(frame, (640, 480))

    # Get the height and width of the frame
    (h, w) = frame.shape[:2]

    # Create a blob from the frame
    #the image that is analyuzed
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

    # Pass the blob through the face detector
    face_detector.setInput(blob)
    detections = face_detector.forward()

    for i in range(0, detections.shape[2]): #the number of faces
        confidence = detections[0, 0, i, 2]
        
        # Set a confidence threshold to filter out low-confidence detections
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x, y, x1, y1) = box.astype("int")

            # Extract the face ROI (Region of Interest)
            face_roi = frame[y:y1, x:x1]

            # Resize the face ROI to match the input shape of the model (48x48)
            resized_face = cv2.resize(face_roi, (48, 48), interpolation=cv2.INTER_AREA)

            # Convert the resized face image to grayscale
            grayscale_face = cv2.cvtColor(resized_face, cv2.COLOR_BGR2GRAY)

            # Normalize the grayscale face image
            normalized_face = grayscale_face / 255.0

            # Reshape the image to match the input shape of the model (1, 48, 48, 1)
            reshaped_face = normalized_face.reshape(1, 48, 48, 1)

            # Predict emotions using the pre-trained model
            preds = model.predict(reshaped_face)[0]
            emotion_idx = preds.argmax()
            emotion = emotion_labels[emotion_idx]


            # Determine satisfaction based on predicted emotion
            if emotion in ['happy', 'surprise']:
                satisfaction = 0.8  # High satisfaction
            elif emotion == 'neutral':
                satisfaction = 0.5  # Neutral satisfaction
            else:
                satisfaction = 0.2  # Low satisfaction

            # Categorize emotions based on rules
            if satisfaction >= 0.6:
                category = 'Satisfied'
            elif satisfaction >= 0.4:
                category = 'Neutral'
            else:
                category = 'Unsatisfied'

            # Get the current timestamp with seconds
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            # Append timestamp, emotion, satisfaction, and category to the DataFrame
            satisScore = {"Satisfied" : 1, "Neutral": .5, "Unsatisfied" : 0}
            
            satisfaction_data['Timestamp'].append(timestamp)
            #satisfaction_data = satisfaction_data.append({'Timestamp': timestamp, 'Satisfaction': satisScore[category]}, ignore_index=True)

            # Draw rectangle around the face and label with predicted emotion and category
            cv2.rectangle(frame, (x, y), (x1, y1), (0, 0, 255), 2)
            cv2.putText(frame, f'{emotion} - {category}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # Display the resulting frame
    cv2.imshow('Real-time Emotion Detection', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close all windows
cap.release()
cv2.destroyAllWindows()

# Export the DataFrame to an Excel file
satisfaction_data.to_excel('satisfaction_data.xlsx', index=False)

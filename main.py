import cv2
import random
import pandas as pd
from deepface import DeepFace
from datetime import datetime  # Import the datetime module

# Load the pre-trained emotion detection model
model = DeepFace.build_model("Emotion")

# Define emotion labels
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

# Load face cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start capturing video
cap = cv2.VideoCapture(0)

# Create an empty DataFrame to store satisfaction data
satisfaction_data = pd.DataFrame(columns=['Timestamp', 'Emotion', 'Satisfaction'])  # Include 'Timestamp' column

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Extract the face ROI (Region of Interest)
        face_roi = gray_frame[y:y + h, x:x + w]

        # Resize the face ROI to match the input shape of the model
        resized_face = cv2.resize(face_roi, (48, 48), interpolation=cv2.INTER_AREA)

        # Normalize the resized face image
        normalized_face = resized_face / 255.0

        # Reshape the image to match the input shape of the model
        reshaped_face = normalized_face.reshape(1, 48, 48, 1)

        # Predict emotions using the pre-trained model
        preds = model.predict(reshaped_face)[0]
        emotion_idx = preds.argmax()
        emotion = emotion_labels[emotion_idx]

        # Assuming satisfaction is a random number between 0 and 1
        satisfaction = random.uniform(0, 1)

        # Get the current timestamp with seconds
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # Append timestamp, emotion, and satisfaction to the DataFrame
        satisfaction_data = satisfaction_data.append({'Timestamp': timestamp, 'Emotion': emotion, 'Satisfaction': satisfaction}, ignore_index=True)

        # Draw rectangle around face and label with predicted emotion
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

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
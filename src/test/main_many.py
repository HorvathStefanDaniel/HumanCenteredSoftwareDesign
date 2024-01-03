import cv2
import numpy as np
import tensorflow as tf
import time
from datetime import datetime
import os
import pyautogui
import csv

# Create the "output/csv" directory if it doesn't exist
csv_output_folder = "output/csv"
if not os.path.exists(csv_output_folder):
    os.makedirs(csv_output_folder)

# Generate a new CSV file name with a timestamp
csv_file = os.path.join(csv_output_folder, f'emotions_data_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.csv')

# Create the "output" folder if it doesn't exist
output_folder = "output"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Get the current timestamp
current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# Get the screen resolution
SCREEN_X, SCREEN_Y = 1920, 1080

# Define the output file paths for webcam and screen recording
webcam_output_file = os.path.join(output_folder, f'webcam_{current_time}.avi')
screen_output_file = os.path.join(output_folder, f'screen_{current_time}.avi')

# Define the codec for webcam and screen recording
webcam_fourcc = cv2.VideoWriter_fourcc(*'XVID')
screen_fourcc = cv2.VideoWriter_fourcc(*'XVID')

# Create VideoWriter objects for webcam and screen recording
frame_rate = 12
webcam_out = cv2.VideoWriter(webcam_output_file, webcam_fourcc, frame_rate, (640, 480))
screen_out = cv2.VideoWriter(screen_output_file, screen_fourcc, frame_rate, (SCREEN_X, SCREEN_Y))

# Load the pre-trained face detection model (Haar Cascade)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load the pre-trained emotion recognition model (HDF5 format)
emotion_model = tf.keras.models.load_model('src/test/model1.hdf5')

# Define emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Initialize the webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, frame_rate)

# Initialize a timer variable for emotion prediction
emotion_timer = time.time()

# Open the CSV file for writing
with open(csv_file, mode='w', newline='') as file:
    fieldnames = ['video_time_readable', 'elapsed_time', 'Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral', 'DetectedString']
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()

    # Initialize the timer variables for frame rate calculation
    start_time = time.time()

    while True:
        ret, webcam_frame = cap.read()
        gray = cv2.cvtColor(webcam_frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

        current_time = time.time()
        elapsed_time = current_time - start_time
        minutes = int(elapsed_time // 60)
        seconds = int(elapsed_time % 60)

        # Initialize default emotion data
        emotion_data = {
            'video_time_readable': f'{minutes:02d}:{seconds:02d}',
            'elapsed_time': elapsed_time,
            'Angry': '', 'Disgust': '', 'Fear': '', 'Happy': '', 'Sad': '', 'Surprise': '', 'Neutral': '', 'DetectedString': ''
        }

        if len(faces) > 0:
            x, y, w, h = faces[0]
            cropped_face = gray[y:y + h, x:x + w]
            cropped_face = cv2.resize(cropped_face, (48, 48))
            cropped_face = cropped_face / 255.0

            if current_time - emotion_timer >= 0.5:
                emotion_timer = current_time
                cropped_face = np.expand_dims(cropped_face, axis=0)
                emotion_probabilities = emotion_model.predict(cropped_face)[0]
                detected_emotion = emotion_labels[np.argmax(emotion_probabilities)]

                # Update emotion data
                emotion_data.update({
                    'Angry': emotion_probabilities[0], 'Disgust': emotion_probabilities[1], 'Fear': emotion_probabilities[2],
                    'Happy': emotion_probabilities[3], 'Sad': emotion_probabilities[4], 'Surprise': emotion_probabilities[5],
                    'Neutral': emotion_probabilities[6], 'DetectedString': detected_emotion
                })

        # Write the data to the CSV file
        writer.writerow(emotion_data)

        # Display the frame
        cv2.imshow('Emotion Detection', webcam_frame)

        # Capture screen frame and convert color channels from RGB to BGR
        screen_frame = np.array(pyautogui.screenshot(region=(0, 0, SCREEN_X, SCREEN_Y)))
        screen_frame = cv2.cvtColor(screen_frame, cv2.COLOR_RGB2BGR)

        # Write frames to output videos
        webcam_out.write(webcam_frame)
        screen_out.write(screen_frame)

        # Exit the loop if 'q' is pressed or if the window is closed
        if cv2.waitKey(1) & 0xFF == ord('q') or elapsed_time > 20:  # 'q' or 20 seconds for testing
            break

# Release the webcam and close all OpenCV windows
webcam_out.release()
screen_out.release()
cap.release()
cv2.destroyAllWindows()

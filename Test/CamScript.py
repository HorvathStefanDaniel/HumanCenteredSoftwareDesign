import cv2
import numpy as np
import tensorflow as tf
import time  # Import the time module

#video recording 
from datetime import datetime
import os  # Import the os module
import pyautogui

#for csv
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
SCREEN_X, SCREEN_Y = 1920, 1080  # Set your screen resolution, maybe we can detect this somehow


# Define the output file paths for webcam and screen recording
webcam_output_file = os.path.join(output_folder, f'webcam_{current_time}.avi')
screen_output_file = os.path.join(output_folder, f'screen_{current_time}.avi')

# Define the codec for webcam and screen recording
webcam_fourcc = cv2.VideoWriter_fourcc(*'XVID')
screen_fourcc = cv2.VideoWriter_fourcc(*'XVID')

# Create VideoWriter objects for webcam and screen recording
webcam_out = cv2.VideoWriter(webcam_output_file, webcam_fourcc, 15.0, (640, 480))  # Adjust frame size as needed
screen_out = cv2.VideoWriter(screen_output_file, screen_fourcc, 15.0, (SCREEN_X, SCREEN_Y))

# Load the pre-trained face detection model (Haar Cascade)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load the pre-trained emotion recognition model (HDF5 format)
emotion_model = tf.keras.models.load_model('model.hdf5')  # Replace with the path to your emotion recognition model

# emotion_model = tf.keras.models.load_model('model-2-cpu.h5')  # Replace with the path to your emotion recognition model
# emotion_model = load_model('model-Ahmadullah.h5')  # Replace with the path to your emotion recognition model

# Define emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

emotion_label = ""

# Initialize the webcam
cap = cv2.VideoCapture(0)  # 0 corresponds to the default camera, change it if necessary
cap.set(cv2.CAP_PROP_FPS, 30)

# Initialize a timer variable for emotion prediction
emotion_timer = time.time()
csv_timer = time.time()

#have to open the csv
with open(csv_file, mode='w', newline='') as file:
     # Define the column names for the CSV file
    fieldnames = ['video_time_readable','elapsed_time', 'Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral', 'DetectedString']
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()

    start_time = time.time()  # Get the start time of the program

    #while loop
    while True:
        # Capture webcam frame
        ret, webcam_frame = cap.read()  
        # Convert the frame to grayscale for face detection
        gray = cv2.cvtColor(webcam_frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

        if len(faces) > 0:
            # Check if enough time has passed since the last emotion prediction
            current_time = time.time()

            # Extract and save the first detected face region
            x, y, w, h = faces[0]
            cropped_face = gray[y:y + h, x:x + w]

            # Resize the cropped face to the input size expected by the emotion recognition model (48x48)
            cropped_face = cv2.resize(cropped_face, (48, 48))

            # Normalize pixel values to be between 0 and 1
            webcam_frame[48:96, 48:96, 1] = (cropped_face * 255).astype(np.uint8)
            cropped_face = cropped_face / 255.0
            webcam_frame[48:96, 0:48, 1] = (cropped_face * 255).astype(np.uint8)

            # Expand the dimensions to match the input shape of the model (add a batch dimension)
            cropped_face = np.expand_dims(cropped_face, axis=-1)

            # Calculate elapsed time in minutes, seconds, and milliseconds
            elapsed_time = time.time() - start_time
            minutes = int(elapsed_time // 60)
            seconds = int(elapsed_time % 60)

            # Predict emotions using the emotion recognition model
            if current_time - emotion_timer >= 0.5:
                emotion_probabilities = emotion_model.predict(np.expand_dims(cropped_face, axis=0))[0]
                emotion_label = emotion_labels[np.argmax(emotion_probabilities)]
                emotion_timer = current_time

                # Create a dictionary with emotion probabilities
                emotion_data = {
                    'video_time_readable' : (str(minutes) + ":" + str(seconds)),
                    'elapsed_time': elapsed_time,
                    'Angry': emotion_probabilities[0],
                    'Disgust': emotion_probabilities[1],
                    'Fear': emotion_probabilities[2],
                    'Happy': emotion_probabilities[3],
                    'Sad': emotion_probabilities[4],
                    'Surprise': emotion_probabilities[5],
                    'Neutral': emotion_probabilities[6],
                    'DetectedString' : emotion_label
                }

                # Write the data to the CSV file
                writer.writerow(emotion_data)

            # Draw a rectangle around the detected face in the original frame
            cv2.rectangle(webcam_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Display the detected emotion as text on the frame
            cv2.putText(webcam_frame, emotion_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Display the cropped face in the top-left corner of the frame
            webcam_frame[0:48, 0:48] = (cropped_face * 255).astype(np.uint8)
        else:
            emotion_label = ""

        # Display the frame
        cv2.imshow('Emotion Detection', webcam_frame)

        # Capture screen frame and convert color channels from RGB to BGR
        screen_frame = np.array(pyautogui.screenshot(region=(0, 0, SCREEN_X, SCREEN_Y)))
        screen_frame = cv2.cvtColor(screen_frame, cv2.COLOR_RGB2BGR)


        # Write frames to output videos
        webcam_out.write(webcam_frame)
        screen_out.write(screen_frame)

        # Exit the loop if 'q' is pressed or if the window is closed
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q') or key == 27:  # 'q' or Esc key
            break

# Release the webcam and close all OpenCV windows
webcam_out.release()
screen_out.release()
cap.release()
cv2.destroyAllWindows()

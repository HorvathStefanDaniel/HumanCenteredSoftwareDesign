import cv2
import numpy as np
import tensorflow as tf
import time  # Import the time module

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

# Initialize a timer variable for emotion prediction
emotion_timer = time.time()

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

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
        frame[48:96, 48:96, 1] = (cropped_face * 255).astype(np.uint8)
        cropped_face = cropped_face / 255.0
        frame[48:96, 0:48, 1] = (cropped_face * 255).astype(np.uint8)

        # Expand the dimensions to match the input shape of the model (add a batch dimension)
        cropped_face = np.expand_dims(cropped_face, axis=-1)

        # Predict emotions using the emotion recognition model
        if current_time - emotion_timer >= 0.5:
            emotion_probabilities = emotion_model.predict(np.expand_dims(cropped_face, axis=0))
            emotion_label = emotion_labels[np.argmax(emotion_probabilities)]
            emotion_timer = current_time

        # Draw a rectangle around the detected face in the original frame
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Display the detected emotion as text on the frame
        cv2.putText(frame, emotion_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Display the cropped face in the top-left corner of the frame
        frame[0:48, 0:48] = (cropped_face * 255).astype(np.uint8)
    else:
        emotion_label = ""

    # Display the frame
    cv2.imshow('Emotion Detection', frame)

    # Exit the loop if 'q' is pressed or if the window is closed
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q') or key == 27:  # 'q' or Esc key
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()

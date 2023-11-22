import torch
import cv2
from torchvision import transforms
from ultralytics import YOLO
import numpy as np
import os
import csv
import time
from datetime import datetime

# Create necessary directories
output_folder = "src/Prototype_1/output"
csv_output_folder = os.path.join(output_folder, "csv")
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
if not os.path.exists(csv_output_folder):
    os.makedirs(csv_output_folder)

# Generate CSV and video output filenames
csv_file_name = f'emotions_data_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.csv'
csv_file = os.path.join(csv_output_folder, csv_file_name)
video_file_name = f'webcam_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.avi'
video_file = os.path.join(output_folder, video_file_name)

# Load models
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
emotion_model = YOLO("src/Prototype_1/best1.pt")

# Video recording setup
fourcc = cv2.VideoWriter_fourcc(*'XVID')
frame_rate = 11
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, frame_rate)
ret, frame = cap.read()
video_out = cv2.VideoWriter(video_file, fourcc, frame_rate, (frame.shape[1], frame.shape[0]))

# CSV setup
fieldnames = ['timestamp', 'emotion']
with open(csv_file, 'w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()

    # Processing loop
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

       

        for (x, y, w, h) in faces:

            #innitialize variables
            xmin, ymin, xmax, ymax = 0, 0, 0, 0
            counter = 0

            # Prepare face image for emotion detection
            face_img = frame[y:y+h, x:x+w]
            transform = transforms.Compose([transforms.ToPILImage(), transforms.Resize((640, 640)), transforms.ToTensor()])
            face_tensor = transform(face_img).unsqueeze(0)
            
            # Emotion detection
            prediction = emotion_model(face_tensor)[0]

            # Process prediction
            if prediction.boxes and len(prediction.boxes):
                for data in prediction.boxes.data.tolist():
                    confidence = data[4]
                    if confidence < 0.85:
                        continue
                    xmin, ymin, xmax, ymax = map(int, data[:4])
                    emotion = prediction.names[int(data[5])]

                    print(data)

                    xmin = int(x + xmin * w / 640)
                    xmax = int(x + xmax * w / 640)
                    ymin = int(y + ymin * h / 640)
                    ymax = int(y + ymax * h / 640)

                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                    cv2.putText(frame, emotion, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                    counter = 20

                    # Write to CSV
                    writer.writerow({
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'emotion': emotion
                    })
            elif counter > 0:
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                cv2.putText(frame, emotion, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                counter -= 1
                

        # Display and record output
        cv2.imshow('Emotion Detection', frame)
        video_out.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Cleanup
video_out.release()
cap.release()
cv2.destroyAllWindows()

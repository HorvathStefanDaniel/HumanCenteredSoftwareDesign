import torch
import cv2
from torchvision import transforms
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('src/Prototype_1/best1.pt')  # Load the model

# Initialize the camera
cap = cv2.VideoCapture(0)

CONFIDENCE_THRESHOLD = 0.8

while True:
    # Capture a frame from the camera
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the image to grayscale and then to RGB
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rgb_image = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

    # Resize and transform the image for the model
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((640, 640)),  # Adjust size if needed for YOLOv8
        transforms.ToTensor(),
    ])
    image = transform(bgr_image)
    
    image = image.unsqueeze(0)  # Add a batch dimension

    # Make a prediction

    prediction = model(image)[0]

    # Check if prediction.boxes exists and is not empty
    if prediction.boxes and len(prediction.boxes):
        for data in prediction.boxes.data.tolist():
            # Extract data from the box
            confidence = data[4]

            # filter out weak detections by ensuring the 
            # confidence is greater than the minimum confidence
            if float(confidence) < CONFIDENCE_THRESHOLD:
                continue
            print(data)

            label = prediction.names[int(data[5])]
            xmin, ymin, xmax, ymax = int(data[0]), int(data[1]), int(data[2]), int(data[3])
            cv2.rectangle(frame, (xmin, ymin) , (xmax, ymax), (0, 255, 0), 2)
            cv2.putText(frame, label, (int(xmin), int(xmax) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    else:
        # If no boxes are detected, display "Nobody detected"
        cv2.putText(frame, "Nobody detected", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display the resulting frame
    cv2.imshow('Emotion Detection', frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close windows
cap.release()
cv2.destroyAllWindows()

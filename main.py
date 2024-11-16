from ultralytics import YOLO
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the YOLOv8 model
model = YOLO('face_yolov8n.pt')  # you need to download this model first

# Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    # Read frame
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv8 inference on the frame
    results = model(frame)
    
    # Visualize the results on the frame
    for result in results:
        boxes = result.boxes.cpu().numpy()
        for box in boxes:
            # Get box coordinates
            x1, y1, x2, y2 = box.xyxy[0].astype(int)
            # Get confidence score
            confidence = box.conf[0]
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Add confidence score
            cv2.putText(frame, f'Face: {confidence:.2f}', 
                       (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, 
                       (0, 255, 0), 
                       2)

    # Display the frame using matplotlib instead of cv2.imshow
    plt.clf()
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    plt.pause(0.001)

    # Break loop with keyboard input
    if plt.waitforbuttonpress(0.001):  # Check for any key press
        break

# Release resources
cap.release()
plt.close()
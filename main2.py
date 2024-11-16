from ultralytics import YOLO
import cv2
import cvzone
import math
import time
import matplotlib.pyplot as plt

# Load YOLOv8 model
model = YOLO('yolov8n.pt')

# Start webcam
cap = cv2.VideoCapture(0)  # Use 0 for default webcam
cap.set(3, 1280)  # Width
cap.set(4, 720)   # Height

# List of class names
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"]

prev_frame_time = 0
new_frame_time = 0

while True:
    success, img = cap.read()
    if not success:
        break
        
    # Get FPS
    new_frame_time = time.time()
    fps = 1/(new_frame_time-prev_frame_time)
    prev_frame_time = new_frame_time
    
    # Perform detection
    results = model(img, stream=True)
    
    # Process detections
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            
            # Class name
            cls = int(box.cls[0])
            currentClass = classNames[cls]
            
            # Draw only if confidence is above 0.5
            if conf > 0.5:
                # Draw bounding box
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
                
                # Draw label background
                cvzone.putTextRect(img, f'{currentClass} {conf}',
                                 (max(0, x1), max(35, y1)),
                                 scale=1,
                                 thickness=1)
    
    # Display FPS
    cv2.putText(img, f'FPS: {int(fps)}', (20, 70), 
                cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)
    
    # Display using matplotlib instead of cv2.imshow
    plt.clf()
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.pause(0.001)
    
    # Break loop with any key press
    if plt.waitforbuttonpress(0.001):
        break

cap.release()
plt.close()
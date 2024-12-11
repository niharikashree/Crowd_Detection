import cv2
from ultralytics import YOLO

model = YOLO('yolov8n.pt')  # Load YOLO model

# Initialize video capture
cap = cv2.VideoCapture(0)

frame_count = 0  # Variable to count frames

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO model on the frame
    results = model(frame, stream=True)

    # Display only once every 30 frames
    if frame_count % 30 == 0:
        person_count = sum(1 for box in results[0].boxes if box.cls == 0 and box.conf > 0.5)  # Class 0 is 'person'
        print(f"People detected: {person_count}")
    
    # Display frame
    cv2.imshow("YOLO Crowd Detection", frame)

    # Exit on pressing ESC
    if cv2.waitKey(1) & 0xFF == 27:
        break

    frame_count += 1

cap.release()
cv2.destroyAllWindows()

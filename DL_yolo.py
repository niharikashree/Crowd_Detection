from ultralytics import YOLO
print("YOLO module imported successfully!")


from ultralytics import YOLO
import cv2

# Load YOLOv8 model
model = YOLO('yolov8n.pt')  # Use 'yolov8n.pt' for a lightweight model

# Initialize the video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO model on the frame
    results = model(frame, stream=True)
    
    # Draw bounding boxes for detected people
    for result in results:
        for box in result.boxes:
            cls = box.cls
            conf = box.conf
            if cls == 0 and conf > 0.5:  # '0' class corresponds to 'person'
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
    
    # Display the frame
    cv2.imshow("YOLO Crowd Detection", frame)
    
    # Exit on pressing ESC
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

import cv2
from ultralytics import YOLO

model = YOLO('yolov8n.pt')  # Load YOLO model

# Threshold for crowd density
ALERT_THRESHOLD = 5

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect people
    results = model(frame, stream=True)
    person_count = 0
    
    for result in results:
        for box in result.boxes:
            cls = box.cls
            conf = box.conf
            if cls == 0 and conf > 0.5:  # '0' class corresponds to 'person'
                person_count += 1
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # Display the count on the frame
    cv2.putText(frame, f"Count: {person_count}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Trigger alert if density exceeds threshold
    if person_count > ALERT_THRESHOLD:
        cv2.putText(frame, "ALERT: High Crowd Density!", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # Show the frame
    cv2.imshow("Crowd Detection", frame)
    
    # Exit on ESC
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

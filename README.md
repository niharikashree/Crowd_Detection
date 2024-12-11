1. density_analysis_crowd_alerts.py:
Uses the YOLO model for real-time crowd detection.
Counts the number of people in a frame and displays the count.
Triggers an alert on the screen if the number of detected people exceeds a predefined threshold (ALERT_THRESHOLD).
Displays bounding boxes for detected people.

2. test_camera.py:
A basic script to test the camera feed.
Displays frames captured by the camera in real-time.
Allows exiting the feed by pressing the ESC key.

3. DL_yolo.py:
Implements YOLO for real-time object detection using a video feed.
Draws bounding boxes around detected objects (focused on the 'person' class).
Lightweight implementation using the YOLOv8 model for crowd detection.

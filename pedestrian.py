from ultralytics import YOLO
import cv2
import time

# Load trained YOLOv8 model
model = YOLO("best.pt")  # Replace with your trained model path

# Open webcam or video
cap = cv2.VideoCapture(0)  # Or use a video file e.g., 'video.mp4'

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Inference
    results = model(frame)

    # Draw results
    annotated_frame = results[0].plot()

    # Show result
    cv2.imshow("Night Vision Pedestrian Detection", annotated_frame)

    # Break on ESC key
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

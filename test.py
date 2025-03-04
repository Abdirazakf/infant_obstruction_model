import cv2
import numpy as np

STREAM_URL = "rtsp://192.168.1.144:8554/cam1?buffer_size=2048&rtsp_transport=tcp"

# Load OpenCV Face Detection Model
prototxt_path = "models/deploy.prototxt"
model_path = "models/res10_300x300_ssd_iter_140000.caffemodel"
net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

cap = cv2.VideoCapture(STREAM_URL, cv2.CAP_FFMPEG)

if not cap.isOpened():
    print(" ERROR: Could not open RTSP stream!")
    exit()

frame_count = 0

while frame_count < 100:  # Capture 100 frames for testing
    success, frame = cap.read()
    
    if not success or frame is None or frame.size == 0:
        print("WARNING: No frame received!")
        continue

    # Face Detection
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    faces_detected = 0
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:  # Confidence threshold
            faces_detected += 1

    print(f" Frame {frame_count} received successfully! Faces detected: {faces_detected}")

    frame_count += 1

cap.release()
print("Test completed, frames received & face detection checked successfully!")

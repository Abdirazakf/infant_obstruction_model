import cv2
import numpy as np

# Use the RTSP Stream Instead of WebRTC
STREAM_URL = "rtsp://192.168.1.144:8554/cam1"

# Load OpenCV's DNN face detector model
prototxt_path = "models/deploy.prototxt"
model_path = "models/res10_300x300_ssd_iter_140000.caffemodel"

net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

# Function to detect landmarks using OpenCV DNN
def detect_landmarks_opencv(frame):
    h, w = frame.shape[:2]

    # Convert frame to blob
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Approximate landmarks
            landmarks = np.array([
                [startX + (endX - startX) // 3, startY + (endY - startY) // 3],
                [startX + 2 * (endX - startX) // 3, startY + (endY - startY) // 3],
                [startX + (endX - startX) // 2, startY + (endY - startY) // 2],
                [startX + (endX - startX) // 3, startY + 2 * (endY - startY) // 3],
                [startX + 2 * (endX - startX) // 3, startY + 2 * (endY - startY) // 3],
            ])

            return landmarks, box

    return None, None  # No face detected

# Face-down detection function
def is_face_down(landmarks):
    return landmarks is None or len(landmarks) < 3  # Assume face-down if no face is detected

# Start capturing the RTSP stream
cap = cv2.VideoCapture(STREAM_URL, cv2.CAP_FFMPEG)

if not cap.isOpened():
    print(f"Error: Could not connect to RTSP stream at {STREAM_URL}")
    exit()

face_detected = False  # Flag to start prediction only when a face is seen

while True:
    success, frame = cap.read()
    
    if not success or frame is None or frame.size == 0:
        print("Warning: No valid frame received. Retrying...")
        continue  # Skip to the next frame

    # Detect landmarks
    landmarks, bbox = detect_landmarks_opencv(frame)

    if landmarks is not None:
        face_detected = True  # Enable predictions once a face is detected

    # If a face has been detected before, start face-down detection
    if face_detected:
        face_down = is_face_down(landmarks)

        # Draw results
        if bbox:
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            for (x, y) in landmarks:
                cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)

            if face_down:
                cv2.putText(frame, "ALERT! Face-Down Detected", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # Show frame
    cv2.imshow("Live RTSP Face-Down Monitor", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

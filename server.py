import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
import paho.mqtt.client as mqtt
import os

app = FaceAnalysis(providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

MQTT_BROKER = "693754a8789c4419b4d760a2653cd86e.s1.eu.hivemq.cloud"
MQTT_PORT = 8883
MQTT_TOPIC = "baby_monitor/obstruction"

mqtt_client = mqtt.Client()
mqtt_client.tls_set()
mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)

stream_url = "rtsp://192.168.1.144:8554/cam1"
cap = cv2.VideoCapture(stream_url)

frame_count = 0
face_detected_recently = False

if not os.path.exists("detections"):
    os.makedirs("detections")

def detect_face_and_position(frame):
    global face_detected_recently

    faces = app.get(frame)
    if len(faces) == 0:
        if face_detected_recently:
            return "No Face Detected", None
        else:
            return None, None

    face_detected_recently = True
    face = faces[0]
    bbox = face.bbox.astype(int)

    landmarks = face.kps  # 5 key facial landmarks
    left_eye, right_eye, nose, left_mouth, right_mouth = landmarks

    if nose[1] > left_eye[1] and nose[1] > right_eye[1]:  
        return "Face Up", bbox
    elif abs(left_eye[1] - right_eye[1]) < 10:  
        return "Side Sleeping", bbox
    else:
        return "Face Down", bbox  

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Ignoring empty frame...")
        continue

    frame_count += 1
    if frame_count % 5 != 0:
        continue

    position, bbox = detect_face_and_position(frame)

    if position == "No Face Detected":
        mqtt_client.publish(MQTT_TOPIC, "No baby detected in crib.")
    elif position == "Face Down":
        mqtt_client.publish(MQTT_TOPIC, "ALERT: Baby is face down!")
    elif position == "Side Sleeping":
        mqtt_client.publish(MQTT_TOPIC, "Baby is sleeping on its side.")
    elif position == "Face Up":
        mqtt_client.publish(MQTT_TOPIC, "Baby is sleeping safely on its back.")

    if bbox is not None:
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
        cv2.putText(frame, position, (bbox[0], bbox[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Save the processed frame
        filename = f"detections/frame_{frame_count}.jpg"
        cv2.imwrite(filename, frame)
        print(f"Saved: {filename}")

cap.release()

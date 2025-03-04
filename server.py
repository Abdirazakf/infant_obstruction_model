import os
import sys
import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
import paho.mqtt.client as mqtt
import time

app = FaceAnalysis(providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

MQTT_BROKER = "693754a8789c4419b4d760a2653cd86e.s1.eu.hivemq.cloud"
MQTT_PORT = 8883
MQTT_TOPIC = "baby_monitor/obstruction"

mqtt_client = mqtt.Client()
mqtt_client.username_pw_set("gp4pi", "Group4pi")
mqtt_client.tls_set()
mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)

stream_url = "rtsp://192.168.1.144:8554/cam1"
cap = cv2.VideoCapture(stream_url)

frame_count = 0
last_face_seen_time = None
face_up_detected = False  # True if Face Up was detected
face_down_alert_sent = False  # Prevents repeated alerts

FACE_MISSING_THRESHOLD = 3  # Seconds to consider Face Down

if not os.path.exists("detections"):
    os.makedirs("detections")

def detect_face_and_position(frame):
    global last_face_seen_time, face_up_detected, face_down_alert_sent

    faces = app.get(frame)
    if len(faces) == 0:
        if face_up_detected and last_face_seen_time and (time.time() - last_face_seen_time > FACE_MISSING_THRESHOLD):
            if not face_down_alert_sent:
                face_down_alert_sent = True  # Prevent spamming
                return "Face Down", None
        return None, None

    last_face_seen_time = time.time()  # Reset timer when face is detected
    face_down_alert_sent = False  # Reset Face Down alert flag when face reappears
    face = faces[0]
    bbox = face.bbox.astype(int)

    landmarks = face.kps  # 5 key facial landmarks
    left_eye, right_eye, nose, left_mouth, right_mouth = landmarks

    if nose[1] > left_eye[1] and nose[1] > right_eye[1]:  
        face_up_detected = True  # Mark face as detected
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

    if position == "Face Down":
        print(" ALERT: Baby is face down!")  # Debugging
        mqtt_client.publish(MQTT_TOPIC, " ALERT: Baby is face down!")
    elif position == "Side Sleeping":
        print("Warning: Baby is sleeping on its side.")  # Debugging
        mqtt_client.publish(MQTT_TOPIC, "Warning: Baby is sleeping on its side.")
    elif position == "Face Up":
        print(" Baby is sleeping safely.")  # Debugging
    elif position is None:
        print(" No baby detected.")  # Debugging

    if bbox is not None:
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
        cv2.putText(frame, position, (bbox[0], bbox[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        filename = f"detections/frame_{frame_count}.jpg"
        cv2.imwrite(filename, frame)
        print(f"Saved: {filename}")

cap.release()

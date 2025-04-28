import os
import time
import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from picamera2 import Picamera2
import paho.mqtt.client as mqtt

#  Load InsightFace model
app = FaceAnalysis(providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.1)

#  MQTT Setup
MQTT_BROKER = "693754a8789c4419b4d760a2653cd86e.s1.eu.hivemq.cloud"
MQTT_PORT = 8883
MQTT_TOPIC = "baby_monitor/obstruction"

mqtt_client = mqtt.Client()
mqtt_client.username_pw_set("gp4pi", "Group4pi")
mqtt_client.tls_set()
mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)

#  Initialize Pi Camera at port 0
picam2 = Picamera2(camera_num=1)
picam2.configure(picam2.create_preview_configuration(main={"size": (640, 480)}))
picam2.start()
time.sleep(1)  # Optional: warm-up time

frame_count = 0
last_face_seen_time = None
face_up_detected = False
face_down_alert_sent = False

FACE_MISSING_THRESHOLD = 2  # Seconds before considering Face Down

# Create directory for saving detections
if not os.path.exists("detections"):
    os.makedirs("detections")

def detect_face_and_position(frame):
    global last_face_seen_time, face_up_detected, face_down_alert_sent

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
    faces = app.get(rgb_frame)

    if len(faces) == 0:
        if face_up_detected and last_face_seen_time and (time.time() - last_face_seen_time > FACE_MISSING_THRESHOLD):
            if not face_down_alert_sent:
                face_down_alert_sent = True
                return "Face Down", None
        return None, None

    last_face_seen_time = time.time()
    face_down_alert_sent = False

    face = faces[0]
    bbox = face.bbox.astype(int)

    landmarks = face.kps
    left_eye, right_eye, nose, left_mouth, right_mouth = landmarks

    if nose[1] > left_eye[1] and nose[1] > right_eye[1]:
        face_up_detected = True
        return "Face Up", bbox
    elif abs(left_eye[1] - right_eye[1]) < 10:
        return "Side Sleeping", bbox
    else:
        return "Face Down", bbox

while True:
    frame = picam2.capture_array()
    if frame is None or np.mean(frame) < 5:
        print("❌ Ignoring empty frame...")
        continue

    frame_count += 1
    if frame_count % 5 != 0:
        continue

    position, bbox = detect_face_and_position(frame)

    if position == "Face Down":
        print(" ALERT: Baby is face down!")
        mqtt_client.publish(MQTT_TOPIC, f" ALERT: Baby is face down! {time.strftime('%Y-%m-%d %H:%M:%S')}")
    elif position == "Side Sleeping":
        print("Warning: Baby is sleeping on its side.")
        mqtt_client.publish(MQTT_TOPIC, f"Warning: Baby is sleeping on its side. {time.strftime('%Y-%m-%d %H:%M:%S')}")
    elif position == "Face Up":
        print(" Baby is sleeping safely.")
    elif position is None:
        print("No baby detected.")

    if bbox is not None:
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
        cv2.putText(frame, position, (bbox[0], bbox[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        filename = f"detections/frame_{frame_count}.jpg"
        cv2.imwrite(filename, frame)
        print(f" Saved: {filename}")

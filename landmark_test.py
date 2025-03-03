import cv2
import numpy as np
import os

prototxt_path = "C://Users/farah/Desktop/infant_obstruction_model/models/deploy.prototxt"
model_path = "C://Users/farah/Desktop/infant_obstruction_model/models/res10_300x300_ssd_iter_140000.caffemodel"

net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

EYE_LANDMARKS = [0, 1]
NOSE_LANDMARK = [2]
MOUTH_LANDMARKS = [3, 4]

# Function to detect landmarks using OpenCV DNN
def detect_landmarks_opencv(image_path):
    image = cv2.imread(image_path)
    h, w = image.shape[:2]

    # Convert image to blob
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            landmarks = np.array([
                [startX + (endX - startX) // 3, startY + (endY - startY) // 3],  # Left Eye
                [startX + 2 * (endX - startX) // 3, startY + (endY - startY) // 3],  # Right Eye
                [startX + (endX - startX) // 2, startY + (endY - startY) // 2],  # Nose
                [startX + (endX - startX) // 3, startY + 2 * (endY - startY) // 3],  # Left Mouth
                [startX + 2 * (endX - startX) // 3, startY + 2 * (endY - startY) // 3],  # Right Mouth
            ])

            return landmarks, image

    return None, image

# Face down detection
def is_face_down(landmarks):
    if landmarks is None:
        return True
    
    visible_landmarks = [landmarks[i] for i in EYE_LANDMARKS + NOSE_LANDMARK + MOUTH_LANDMARKS if i < len(landmarks)]
    
    return len(visible_landmarks) < 3  # If less than 3 key points detected assume face down


test_image_path = "C://Users/farah/Desktop/infant_obstruction_model/dataset/fd/youtube-06.png"
landmarks, output_image = detect_landmarks_opencv(test_image_path)

face_down = is_face_down(landmarks)
print(f"Face-Down Prediction: {face_down}")

if landmarks is not None:
    # Draw landmarks
    for (x, y) in landmarks:
        cv2.circle(output_image, (x, y), 2, (0, 0, 255), -1)

    # Display image
    cv2.imshow("Face Landmarks", output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
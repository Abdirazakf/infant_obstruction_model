import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

dataset_path = "/home/grp4pi/AIFiles/infant_obstruction_model/dataset/labels.csv"
image_folder = "/home/grp4pi/AIFiles/infant_obstruction_model/dataset/images/"

df = pd.read_csv(dataset_path)
print(df.head())

# Function to determine if face is down based on landmark
def is_face_down(landmarks):
    eye_indices = [36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47]
    nose_indices = [27, 28, 29, 30]
    mouth_indices = [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59]
    visible_landmarks = [landmarks[i] for i in eye_indices + nose_indices + mouth_indices if i < len(landmarks)]
    
    # If key facial features are missing, classify as face down
    return len(visible_landmarks) < (len(eye_indices) + len(nose_indices) + len(mouth_indices)) * 0.5

# Function to plot landmarks on an image
def plot_landmarks(image_path, landmarks, face_down):
    if not os.path.exists(image_path):
        print(f"Error: Image file {image_path} not found.")
        return
    
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Unable to load image {image_path}.")
        return
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    plt.figure(figsize=(5, 5))
    plt.imshow(img)

    landmarks = np.array(landmarks).reshape(-1, 2)
    # Plot landmarks
    plt.scatter(landmarks[:, 0], landmarks[:, 1], c='red', marker='x')
    
    title = "Face-Down" if face_down else "Face-Up"
    plt.title(title)
    plt.show()

num_samples = 410
for idx in range(min(num_samples, len(df))):
    sample = df.iloc[idx]
    image_subfolder = sample["image-set"].strip()
    target_folder = os.path.join(image_folder, image_subfolder)
    image_filename = sample["filename"].strip()
    image_path = os.path.join(target_folder, image_filename)
    image_path = os.path.normpath(image_path)
    
    # Extract landmark coordinates
    landmark_columns = [col for col in df.columns if col.startswith("gt-x") or col.startswith("gt-y")]
    landmarks = sample[landmark_columns].values.astype(float).reshape(-1, 2)

    face_down = is_face_down(landmarks)
    print(f"Image: {image_filename}, Face-Down: {face_down}")
    
    if face_down:
        plot_landmarks(image_path, landmarks, face_down)

import os  # Importing the os module to handle directory and file path operations
import pickle  # Importing pickle to save the processed data in a serialized format
import mediapipe as mp  # Importing mediapipe for hand landmark detection
import cv2  # Importing OpenCV for image processing
import matplotlib.pyplot as plt  # Importing matplotlib for potential visualization (though not used in this script)

# Initializing Mediapipe components for hand detection and drawing utilities
mp_hands = mp.solutions.hands  # Access the hand solution module from Mediapipe
mp_drawing = mp.solutions.drawing_utils  # Drawing utility to visualize hand landmarks
mp_drawing_styles = mp.solutions.drawing_styles  # Drawing styles to style landmarks and connections

# Setting up the Mediapipe hand detector with specific configurations
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)
# `static_image_mode=True`: Treat input images as standalone images (no smoothing).
# `min_detection_confidence=0.3`: Minimum confidence value for hand detection to be considered successful.

# Directory where the image data for each class is stored
DATA_DIR = './data'

# Initialize lists to store the data and labels for each image
data = []  # List to hold the processed hand landmark data for each image
labels = []  # List to hold the corresponding labels (class) for each image

# Iterate through each class directory in the `DATA_DIR`
for dir_ in os.listdir(DATA_DIR):
    # For each class directory, iterate through the images
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        data_aux = []  # Temporary list to store normalized hand landmark data for a single image
        x_ = []  # List to store x-coordinates of hand landmarks
        y_ = []  # List to store y-coordinates of hand landmarks

        # Read the image using OpenCV
        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))  # Read the image from the file path
        img_rgb = cv2.cvtColor(img,
                               cv2.COLOR_BGR2RGB)  # Convert the image from BGR (OpenCV format) to RGB (Mediapipe format)

        # Process the image using Mediapipe to find hand landmarks
        results = hands.process(img_rgb)

        # If hand landmarks are detected in the image
        if results.multi_hand_landmarks:
            # For each hand detected in the image (though typically only one hand is expected)
            for hand_landmarks in results.multi_hand_landmarks:
                # Loop through each landmark point in the hand
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x  # Get the x-coordinate of the landmark
                    y = hand_landmarks.landmark[i].y  # Get the y-coordinate of the landmark

                    # Append the landmark coordinates to x_ and y_ lists
                    x_.append(x)
                    y_.append(y)

                # Normalize the hand landmarks by subtracting the minimum x and y values
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x  # Re-fetch the x-coordinate
                    y = hand_landmarks.landmark[i].y  # Re-fetch the y-coordinate

                    # Normalize each landmark coordinate relative to the minimum x and y values
                    data_aux.append(x - min(x_))  # Store the normalized x-coordinate
                    data_aux.append(y - min(y_))  # Store the normalized y-coordinate

            # Append the processed data for this image to the main `data` list
            data.append(data_aux)
            # Append the corresponding label (directory name) to the `labels` list
            labels.append(dir_)

# Open a file in write-binary mode to save the processed data and labels
f = open('data.pickle', 'wb')
# Use pickle to serialize and save the data and labels into the file
pickle.dump({'data': data, 'labels': labels}, f)
# Close the file after saving
f.close()

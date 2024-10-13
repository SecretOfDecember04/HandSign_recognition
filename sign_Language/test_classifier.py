import pickle  # Importing the pickle module to load the pre-trained model from a file

import cv2  # Importing OpenCV for video capture and image processing
import mediapipe as mp  # Importing Mediapipe for hand landmark detection
import numpy as np  # Importing numpy for numerical operations on the data

# Load the pre-trained model stored in 'model.p' using pickle
model_dict = pickle.load(open('./model.p', 'rb'))  # Load the serialized model dictionary from a file
model = model_dict['model']  # Extract the trained model from the dictionary

# Initialize video capture from the default camera (index 0)
cap = cv2.VideoCapture(0)  # Open the video capture stream from the primary camera

# Set up Mediapipe components for hand tracking and drawing
mp_hands = mp.solutions.hands  # Access the hand solution module from Mediapipe
mp_drawing = mp.solutions.drawing_utils  # Drawing utility to visualize hand landmarks
mp_drawing_styles = mp.solutions.drawing_styles  # Drawing styles to style landmarks and connections

# Create a hands object with specific configurations
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)
# `static_image_mode=True`: Treat input frames as independent images.
# `min_detection_confidence=0.3`: Minimum confidence value for the hand detection to be successful.

# Define a dictionary to map model predictions to labels
labels_dict = {0: 'A', 1: 'B', 2: 'L'}  # Map the class indices to hand gesture labels

# Start an infinite loop to process video frames in real-time
while True:

    # Initialize lists to store hand landmark data
    data_aux = []  # List to store normalized landmark data for the detected hand
    x_ = []  # List to store x-coordinates of hand landmarks
    y_ = []  # List to store y-coordinates of hand landmarks

    # Capture a single frame from the video stream
    ret, frame = cap.read()  # `ret` is True if the frame was captured successfully

    # Get the height and width of the captured frame
    H, W, _ = frame.shape  # `H` is height, `W` is width, `_` is the number of color channels

    # Convert the captured frame from BGR (OpenCV format) to RGB (Mediapipe format)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the RGB frame using Mediapipe to detect hand landmarks
    results = hands.process(frame_rgb)

    # Check if any hand landmarks were detected
    if results.multi_hand_landmarks:
        # If hand landmarks are found, iterate through each hand (though only one hand is expected here)
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw the landmarks and connections on the original frame
            mp_drawing.draw_landmarks(
                frame,  # The frame to draw on
                hand_landmarks,  # Detected hand landmarks
                mp_hands.HAND_CONNECTIONS,  # Connections between landmarks
                mp_drawing_styles.get_default_hand_landmarks_style(),  # Landmark style
                mp_drawing_styles.get_default_hand_connections_style())  # Connection style

        # Collect the x and y coordinates of each landmark
        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x  # Get x-coordinate of the landmark
                y = hand_landmarks.landmark[i].y  # Get y-coordinate of the landmark

                # Append the landmark coordinates to the x_ and y_ lists
                x_.append(x)
                y_.append(y)

            # Normalize the hand landmarks based on the minimum x and y values
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x  # Re-fetch the x-coordinate
                y = hand_landmarks.landmark[i].y  # Re-fetch the y-coordinate
                data_aux.append(x - min(x_))  # Store the normalized x-coordinate
                data_aux.append(y - min(y_))  # Store the normalized y-coordinate

        # Calculate the bounding box around the detected hand using normalized coordinates
        x1 = int(min(x_) * W) - 10  # Left corner x-coordinate of the bounding box
        y1 = int(min(y_) * H) - 10  # Top corner y-coordinate of the bounding box

        x2 = int(max(x_) * W) - 10  # Right corner x-coordinate of the bounding box
        y2 = int(max(y_) * H) - 10  # Bottom corner y-coordinate of the bounding box

        # Use the trained model to predict the hand gesture based on the landmark data
        prediction = model.predict([np.asarray(data_aux)])  # Convert `data_aux` to a numpy array and make a prediction

        # Get the predicted character from the label dictionary
        predicted_character = labels_dict[int(prediction[0])]  # Get the label corresponding to the predicted class

        # Draw a rectangle around the detected hand
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)  # Draw the rectangle using the calculated coordinates

        # Display the predicted character above the bounding box
        cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

    # Show the frame with the visualized hand landmarks and prediction
    cv2.imshow('frame', frame)

    # Wait for 1 millisecond and check if the user presses the 'q' key to exit
    cv2.waitKey(1)

# Release the video capture resource and close all OpenCV windows
cap.release()  # Release the camera resource
cv2.destroyAllWindows()  # Close all the OpenCV windows that were opened

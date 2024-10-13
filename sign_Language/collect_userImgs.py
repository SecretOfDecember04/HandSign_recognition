import os  # Importing the os module to interact with the operating system for file and directory operations

import cv2  # Importing OpenCV for video capture and image processing

# Directory where the collected images will be stored
DATA_DIR = './data'

# Check if the directory exists, if not, create it
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Define the number of classes and the number of images to be collected per class
number_of_classes = 3  # Number of classes/categories for which data will be collected
dataset_size = 100  # Number of images to be collected for each class

# Initialize video capture with camera index 1
cap = cv2.VideoCapture(1)  # Opens the video capture stream from the camera with index 1

# Loop through each class to collect data
for j in range(number_of_classes):

    # Create a directory for each class if it doesn't already exist
    if not os.path.exists(os.path.join(DATA_DIR, str(j))):
        os.makedirs(os.path.join(DATA_DIR, str(j)))

    # Print a message indicating which class is being collected
    print('Collecting the data for class {}'.format(j))

    done = False  # Flag to control when to start collecting data

    # Loop to wait until the user is ready to start data collection
    while True:
        ret, frame = cap.read()  # Capture a frame from the camera
        # Display a message on the frame prompting the user to press 'Q' to start data collection
        cv2.putText(frame, 'You Ready? Press "Q" ! :)', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
                    cv2.LINE_AA)
        cv2.imshow('frame', frame)  # Show the frame with the message
        if cv2.waitKey(25) == ord('q'):  # If 'q' is pressed, break the loop to start collecting data
            break

    # Initialize a counter to track the number of images captured for this class
    counter = 0

    # Collect the specified number of images for the current class
    while counter < dataset_size:
        ret, frame = cap.read()  # Capture a new frame from the camera
        cv2.imshow('frame', frame)  # Display the captured frame
        cv2.waitKey(25)  # Add a delay of 25 milliseconds before capturing the next frame

        # Save the captured frame to the corresponding class directory with the counter as the filename
        cv2.imwrite(os.path.join(DATA_DIR, str(j), '{}.jpg'.format(counter)), frame)

        # Increment the counter to track the number of images saved
        counter += 1

# Release the video capture resource and close all OpenCV windows
cap.release()  # Release the camera resource
cv2.destroyAllWindows()  # Close all the OpenCV windows that were opened

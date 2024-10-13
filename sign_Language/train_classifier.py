import pickle  # Importing the pickle module for saving and loading serialized data and models

from sklearn.ensemble import RandomForestClassifier  # Importing Random Forest classifier from scikit-learn
from sklearn.model_selection import train_test_split  # Importing train_test_split for splitting data into training and testing sets
from sklearn.metrics import accuracy_score  # Importing accuracy_score to evaluate the classification performance
import numpy as np  # Importing numpy for efficient numerical operations and array handling


# Load the preprocessed data and labels from the pickle file
data_dict = pickle.load(open('./data.pickle', 'rb'))  # Open and load the serialized data from 'data.pickle'

# Extract data and labels from the loaded dictionary
data = np.asarray(data_dict['data'])  # Convert the 'data' list to a numpy array
labels = np.asarray(data_dict['labels'])  # Convert the 'labels' list to a numpy array

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)
# `test_size=0.2`: 20% of the data will be used for testing, and 80% for training
# `shuffle=True`: Shuffle the data before splitting
# `stratify=labels`: Ensure the class distribution is the same in both train and test sets

# Initialize the Random Forest Classifier
model = RandomForestClassifier()  # Create an instance of the Random Forest classifier

# Train the model using the training data
model.fit(x_train, y_train)  # Fit the Random Forest model on the training data and labels

# Use the trained model to make predictions on the test set
y_predict = model.predict(x_test)  # Predict the labels for the test set

# Calculate the accuracy of the model's predictions
score = accuracy_score(y_predict, y_test)  # Compare the predicted labels with the true labels to get accuracy

# Print the classification accuracy in percentage format
print('{}% of samples were classified correctly !'.format(score * 100))  # Output the accuracy percentage

# Save the trained model to a pickle file for future use
f = open('model.p', 'wb')  # Open a new file in write-binary mode to save the model
pickle.dump({'model': model}, f)  # Serialize and save the trained model in the file
f.close()  # Close the file after saving

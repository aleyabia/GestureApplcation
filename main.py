"""
Created on Thu Jan 28 00:44:25 2021

@author: chakati
"""
import cv2
import csv
import numpy as np
import os
import tensorflow as tf
## import the handfeature extractor class
from handshape_feature_extractor import HandShapeFeatureExtractor  

from sklearn.metrics.pairwise import cosine_similarity

## import the frame extractor class
from frameextractor import frameExtractor

# =============================================================================
# Get the penultimate layer for trainig data
# =============================================================================
# your code goes here
# Extract the middle frame of each gesture video

# Define the paths to your training videos and where you want to save the frames
training_videos_path = 'test'
frames_path = 'training_frames'
if not os.path.exists(frames_path):
    os.makedirs(frames_path)
# Create an instance of the HandShapeFeatureExtractor class
hand_shape_extractor = HandShapeFeatureExtractor.get_instance()

# Initialize an empty list to store the feature vectors
feature_vectors = []

count = 0
# Iterate through all the training videos
for video_filename in os.listdir(training_videos_path):
    if video_filename.endswith('.mp4'):
        video_path = os.path.join(training_videos_path, video_filename)

        # Extract the middle frame from the video
        frameExtractor(video_path, frames_path, count)

        # Increment count for the next frame
        count += 1

        # Load the extracted frame
        frame = cv2.imread(os.path.join(frames_path, "%#05d.png" % (count)), cv2.IMREAD_GRAYSCALE)

        # Extract the hand shape feature
        feature_vector = hand_shape_extractor.extract_feature(frame)

        # Append the feature vector to the list
        feature_vectors.append(feature_vector)

# Now feature_vectors contains the penultimate layer of the training set
# You can save it or use it for further processing, e.g., training a gesture recognition model

# =============================================================================
# Get the penultimate layer for test data
# =============================================================================
# your code goes here 
# Extract the middle frame of each gesture video

# Define the paths to your test videos and where you want to save the frames
test_videos_path = 'traindata'
test_frames_path = 'test_frames'
if not os.path.exists(test_frames_path):
    os.makedirs(test_frames_path)


# Create an instance of the HandShapeFeatureExtractor class
hand_shape_extractor = HandShapeFeatureExtractor.get_instance()

# Initialize an empty list to store the feature vectors for test videos
test_feature_vectors = []

count = 0
# Iterate through all the test videos
for video_filename in os.listdir(test_videos_path):
    if video_filename.endswith('.mp4'):
        video_path = os.path.join(test_videos_path, video_filename)

        # Extract the middle frame from the test video
        frameExtractor(video_path, test_frames_path, count)

        # Increment count for the next frame
        count += 1

        # Load the extracted test frame
        frame = cv2.imread(os.path.join(test_frames_path, "%#05d.png" % (count)), cv2.IMREAD_GRAYSCALE)

        # Extract the hand shape feature for the test frame
        test_feature_vector = hand_shape_extractor.extract_feature(frame)

        # Append the feature vector to the list for test videos
        test_feature_vectors.append(test_feature_vector)

# Now test_feature_vectors contains the penultimate layer of the test dataset
# You can use these vectors for gesture recognition in Task 3
# Convert the list of test feature vectors to a NumPy array
test_feature_vectors = np.array(test_feature_vectors)


# =============================================================================
# Recognize the gesture (use cosine similarity for comparing the vectors)
# =============================================================================
# Load the penultimate layer of the training set (feature_vectors from Task 1)
# ... Your previous code ...
# Convert the list of feature vectors to a NumPy array
training_feature_vectors = np.array(feature_vectors)

# Save the training feature vectors as a .npy file in the same directory as main.py
np.save('training_feature_vectors.npy', training_feature_vectors)
# Add this code to save the training feature vectors as a .npy file
# Load the penultimate layer of the training set (feature_vectors from Task 1)
# Replace this with the actual feature_vectors from Task 1
training_feature_vectors = np.load('training_feature_vectors.npy')  # Load the saved training feature vectors

# Reshape the feature vectors if they have an extra dimension
training_feature_vectors = training_feature_vectors.reshape(training_feature_vectors.shape[0], -1)
test_feature_vectors = test_feature_vectors.reshape(test_feature_vectors.shape[0], -1)


label_mapping = [
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "10",
    "11",
    "12",
    "13",
    "14",
    "15",
    "16"
]

# Initialize an empty list to store the recognized gesture labels
recognized_labels = []


# Iterate through the feature vectors of test videos
for test_feature_vector in test_feature_vectors:
    # Calculate cosine similarity between the test vector and all training vectors
    similarities = cosine_similarity([test_feature_vector], training_feature_vectors)

    # Find the index of the training sample with the highest similarity
    recognized_index = np.argmax(similarities)

    # Assign a recognized label based on the index (assuming you have a mapping)
    
    # Check if the recognized_index is within the valid range of label_mapping
    if 0 <= recognized_index < len(label_mapping):
        recognized_label = label_mapping[recognized_index]
    else:
        recognized_label = "Unknown"  # Handle cases where the index is out of range

    # Append the recognized label to the list
    recognized_labels.append(recognized_label)

# Save the recognized gesture labels vertically in a CSV file
with open('Results.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    
    # Write the labels vertically
    for recognized_label in recognized_labels:
        writer.writerow([recognized_label])


# Add the following code to save the training feature vectors as a .npy file
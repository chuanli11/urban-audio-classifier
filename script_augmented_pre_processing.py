import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import pickle
from include import helpers
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

# Set your path to the dataset
us8k_path = os.path.abspath('./UrbanSound8K')
audio_path = os.path.join(us8k_path, 'audio')
augmented_path = os.path.join(audio_path, 'augmented')

# Metadata
metadata_augmented_path = os.path.abspath('data/augmented-data.csv')


# Load the metadata from the generated CSV
metadata = pd.read_csv(metadata_augmented_path)

# Examine dataframe
print("Metadata length:", len(metadata))
metadata.tail()


# Iterate through all audio files and extract MFCC
features = []
labels = []
frames_max = 0
counter = 0
total_samples = len(metadata)
n_mfcc = 40

for index, row in metadata.iterrows():
    file_path = os.path.join(os.path.abspath(audio_path), 'fold' + str(row["fold"]), str(row["file"]))
    class_label = row["class"]

    # Extract MFCCs (do not add padding)
    mfccs = helpers.get_mfcc(file_path, 0, n_mfcc)
    
    # Save current frame count
    num_frames = mfccs.shape[1]
    
    # Add row (feature / label)
    features.append(mfccs)
    labels.append(class_label)

    # Update frames maximum
    if (num_frames > frames_max):
        frames_max = num_frames
        
    print("Progress: {}/{}".format(index+1, total_samples))
    print("Last file: ", file_path)

    counter += 1
    
print("Finished: {}/{}".format(index, total_samples))


padded = []

# Add padding
mfcc_max_padding = frames_max
for i in range(len(features)):
    size = len(features[i][0])
    if (size < mfcc_max_padding):
        pad_width = mfcc_max_padding - size
        px = np.pad(features[i], 
                    pad_width=((0, 0), (0, pad_width)), 
                    mode='constant', 
                    constant_values=(0,))
    
    padded.append(px)

# Convert features (X) and labels (y) to Numpy arrays

X = np.array(padded)
y = np.array(labels)

# Optionally save the features to disk
np.save("data/X-mfcc-augmented", X)
np.save("data/y-mfcc-augmented", y)

# Verify shapes
print("Raw features length: {}".format(len(features)))
print("Padded features length: {}".format(len(padded)))
print("Feature labels length: {}".format(len(features)))
print("X: {}, y: {}".format(X.shape, y.shape))



# Iterate through all audio files and extract Log-Mel Spectrograms
features = []
labels = []
frames_max = 0
counter = 0
total_samples = len(metadata)
n_mels = 40

for index, row in metadata.iterrows():
    file_path = os.path.join(os.path.abspath(audio_path), 'fold' + str(row["fold"]), str(row["file"]))
    class_label = row["class"]

    # Extract Log-Mel Spectrograms (do not add padding)
    mels = helpers.get_mel_spectrogram(file_path, 0, n_mels=n_mels)
    
    # Save current frame count
    num_frames = mels.shape[1]
    
    # Add row (feature / label)
    features.append(mels)
    labels.append(class_label)

    # Update frames maximum
    if (num_frames > frames_max):
        frames_max = num_frames
        
    print("Progress: {}/{}".format(index+1, total_samples))
    print("Last file: ", file_path)

    counter += 1
    
print("Finished: {}/{}".format(index, total_samples))


padded = []

# Add padding
mels_max_padding = frames_max
for i in range(len(features)):
    size = len(features[i][0])
    if (size < mels_max_padding):
        pad_width = mels_max_padding - size
        px = np.pad(features[i], 
                    pad_width=((0, 0), (0, pad_width)), 
                    mode='constant', 
                    constant_values=(0,))
    
    padded.append(px)


# Convert features (X) and labels (y) to Numpy arrays

X = np.array(padded)
y = np.array(labels)

# Optionally save the features to disk
np.save("data/X-mel_spec-augmented", X)
np.save("data/y-mel_spec-augmented", y)


# Verify shapes
print("Raw features length: {}".format(len(features)))
print("Padded features length: {}".format(len(padded)))
print("Feature labels length: {}".format(len(features)))
print("X: {}, y: {}".format(X.shape, y.shape))
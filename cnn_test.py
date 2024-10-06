import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import scipy.signal
from fractions import Fraction

# Define global parameters
short_window = 512  # ~512 samples around the seismic event, each treated as an input sequence
half_window = short_window // 2
fs_target = 6.625  # Sampling frequency for inference
fs_original = 20  # Original sampling frequency of the test data

# Define bandpass filter (if necessary for preprocessing the input signals)
sos4hz = scipy.signal.butter(4, (3.9, 4.1), btype='bandpass', analog=False, output='sos', fs=fs_original)
sos2_3hz = scipy.signal.butter(4, (2.3, 2.7), btype='bandpass', analog=False, output='sos', fs=fs_original)

def apply_filters(data):
    #filtered_4hz = scipy.signal.sosfilt(sos4hz, data)
    #filtered_2_3hz = scipy.signal.sosfilt(sos2_3hz, data)
    #return filtered_4hz + filtered_2_3hz
    return data

# Resample function using integer up/down ratio
def resample(data, original_fs, target_fs):
    ratio = Fraction(target_fs / original_fs).limit_denominator()
    up = ratio.numerator
    down = ratio.denominator
    return scipy.signal.resample_poly(data, up, down)

# Function to extract samples
def extract_sample(data, target_idx, apply_filter=True, normalize=True):
    """
    Extracts a 'short window' of data centered around target_idx for CNN input.
    Returns a 1D signal array.
    """
    # Define the range for the window (half before and half after the target index)
    start_idx = target_idx - half_window
    end_idx = target_idx + half_window

    # Ensure indices are within the bounds of the data
    if start_idx < 0:
        start_idx = 0
    if end_idx > len(data):
        end_idx = len(data)

    short_win_data = data.iloc[start_idx:end_idx]['velocity(c/s)'].values

    # Optionally normalize the data (Z-score normalization)
    if normalize:
        mean = np.mean(short_win_data)
        std = np.std(short_win_data)
        if std != 0:  # To avoid division by zero
            short_win_data = (short_win_data - mean) / std

    # If the extracted window is shorter than expected (due to being at the start/end), pad it
    if len(short_win_data) < short_window:
        padding = np.zeros(short_window - len(short_win_data))
        short_win_data = np.concatenate([short_win_data, padding])

    return short_win_data

class SeismicCNN(nn.Module):
    def __init__(self):
        super(SeismicCNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 8, kernel_size=5, stride=1, padding=2)  # 8 filters
        self.bn1 = nn.BatchNorm1d(8)  # Batch normalization
        self.conv2 = nn.Conv1d(8, 16, kernel_size=5, stride=1, padding=2)  # 16 filters
        self.bn2 = nn.BatchNorm1d(16)  # Batch normalization
        self.conv3 = nn.Conv1d(16, 32, kernel_size=5, stride=1, padding=2)  # 32 filters
        self.pool = nn.MaxPool1d(2, 2)
        self.dropout = nn.Dropout(p=0.3)
        self.fc1 = nn.Linear(32 * 64, 32)  # 32 channels, 64 samples after pooling
        self.fc2 = nn.Linear(32, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # Flatten the output to (batch_size, 2048)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x
    
# Load the trained model
model = SeismicCNN()
model.load_state_dict(torch.load("seismic_cnn_model.pth"))  # Load the trained model
model.eval()

# Function to perform inference on a single .csv file
def run_inference_on_csv(file_path):
    data = pd.read_csv(file_path)
    
    filtered_data = apply_filters(data['velocity(c/s)'].values)
    
    # Resample the 'velocity(c/s)' data from original_fs to target_fs
    resampled_v = resample(filtered_data, fs_original, fs_target)
    resampled_data = pd.DataFrame({'velocity(c/s)': resampled_v})
    
    # Starting from a delay to avoid predicting too early
    start_delay = 8 * short_window  # Delay inference until some data is processed
    
    for i in range(start_delay, len(resampled_data) - short_window):
        # Extract sample data for the CNN
        sample = extract_sample(resampled_data, i)
        sample_tensor = torch.tensor(sample, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, 512)

        # Run inference
        with torch.no_grad():
            output = model(sample_tensor)
            event_detected = torch.sigmoid(output).item()
            #print(f"Sample {i}: Seismic Event Probability: {event_detected:.4f}")
            if event_detected > 0.9:
                print("Seismic event detected!")
                return i + short_window // 2  # Return the midpoint of the window where the event is detected
    
    # If no event is detected, return None
    return None

# Function to plot the seismic activity
def plot_seismic_activity(file_path, event_time_idx):
    data = pd.read_csv(file_path)
    
    # Plot the seismic activity
    plt.figure(figsize=(10, 6))
    plt.plot(data['rel_time(sec)'], data['velocity(c/s)'], label='Seismic Activity')
    
    # Plot the detected event time with a red vertical line
    if event_time_idx is not None:
        event_time = data.iloc[event_time_idx]['rel_time(sec)']
        plt.axvline(x=event_time, color='red', linestyle='--', label='Detected Event')

    plt.title(f'Seismic Activity from {os.path.basename(file_path)}')
    plt.xlabel('Time (sec)')
    plt.ylabel('Velocity (m/s)')
    plt.legend()
    
    # Save the figure to a file
    output_file = os.path.splitext(file_path)[0] + "_seismic_plot.png"
    plt.savefig(output_file)
    plt.close()
    print(f"Plot saved to {output_file}")

def plot_spectrogram(file_path):
    data = pd.read_csv(file_path)
    velocity = data['velocity(c/s)'].values

    f, t, Sxx = scipy.signal.spectrogram(velocity, fs=fs_original, nperseg=256)
    
    plt.figure(figsize=(10, 6))
    plt.pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud')
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Time (sec)')
    plt.title(f'Spectrogram of {os.path.basename(file_path)}')
    
    output_file = os.path.splitext(file_path)[0] + "_spectrogram.png"
    plt.savefig(output_file)
    plt.close()
    print(f"Spectrogram saved to {output_file}")

# Directory with test .csv files
test_folder = os.path.expanduser("~/Downloads/space_apps_2024_seismic_detection/data/mars/test/data/")

# Iterate through the folder and process .csv files
for file_name in os.listdir(test_folder):
    if file_name.endswith('.csv'):
        file_path = os.path.join(test_folder, file_name)
        
        # Run inference on the file
        event_time_idx = run_inference_on_csv(file_path)
        
        # Plot the results
        plot_seismic_activity(file_path, event_time_idx)
        
        #plot_spectrogram(file_path)



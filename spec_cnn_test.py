import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import scipy.signal
from fractions import Fraction

# Define global parameters
short_window = 512  # ~512 samples around the seismic event, each treated as an input sequence
half_window = short_window // 2
fs_target = 6.625  # Sampling frequency for inference
fs_original = 20  # Original sampling frequency of the test data
fixed_size = (128, 128)  # Target spectrogram size for compatibility with the model

# Resample function using integer up/down ratio
def resample(data, original_fs, target_fs):
    ratio = Fraction(target_fs / original_fs).limit_denominator()
    up = ratio.numerator
    down = ratio.denominator
    return scipy.signal.resample_poly(data, up, down)

# Function to generate a fixed-size spectrogram from data
def extract_spectrogram(data, target_idx, nperseg=128, noverlap=64, normalize=True):
    """
    Generates a spectrogram from a window of seismic data, ensuring it's padded/truncated to fixed size.
    """
    start_idx = target_idx - half_window
    end_idx = target_idx + half_window

    # Ensure indices are within bounds
    if start_idx < 0:
        start_idx = 0
    if end_idx > len(data):
        end_idx = len(data)

    short_win_data = data.iloc[start_idx:end_idx]['velocity(c/s)'].values

    # Generate spectrogram using scipy
    f, t, Sxx = scipy.signal.spectrogram(short_win_data, fs=fs_target, nperseg=nperseg, noverlap=noverlap)
    
    if normalize:
        Sxx = (Sxx - np.mean(Sxx)) / np.std(Sxx)  # Normalize the spectrogram

    # Ensure the spectrogram is of the fixed size
    Sxx_resized = np.zeros(fixed_size)  # Create an empty array of the target size (128x128)
    
    # Get current spectrogram dimensions
    curr_f, curr_t = Sxx.shape
    
    # Fill or truncate to match the fixed size
    Sxx_resized[:min(curr_f, fixed_size[0]), :min(curr_t, fixed_size[1])] = Sxx[:min(curr_f, fixed_size[0]), :min(curr_t, fixed_size[1])]

    return Sxx_resized

# Define the CNN model for 2D spectrogram inputs
class SeismicSpectrogramCNN(nn.Module):
    def __init__(self):
        super(SeismicSpectrogramCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 16 * 16, 64)  # Adjust size based on spectrogram dimensions
        self.fc2 = nn.Linear(64, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load the trained model for spectrograms
model = SeismicSpectrogramCNN()
model.load_state_dict(torch.load("spec_cnn_model.pth"))  # Load the trained model
model.eval()

# Function to perform inference on a single .csv file using spectrograms
def run_inference_on_csv(file_path):
    data = pd.read_csv(file_path)
    
    # Resample the 'velocity(c/s)' data from original_fs to target_fs
    resampled_v = resample(data['velocity(c/s)'].values, fs_original, fs_target)
    resampled_data = pd.DataFrame({'velocity(c/s)': resampled_v})
    
    start_delay = 8 * short_window  # Delay inference until enough data is available
    detected_events = []  # Store all detected seismic events
    
    for i in range(start_delay, len(resampled_data) - short_window):
        # Extract spectrogram data for the CNN
        spectrogram = extract_spectrogram(resampled_data, i)
        spectrogram_tensor = torch.tensor(spectrogram, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, F, T)

        # Run inference
        with torch.no_grad():
            output = model(spectrogram_tensor)
            event_detected = torch.sigmoid(output).item()
            # Print detected probability
            #print(f"Sample {i}: Seismic Event Probability: {event_detected:.4f}")
            if event_detected > 0.25:
                detected_events.append(i + short_window // 2)  # Store all detected event indices
    
    # Return all detected events
    return detected_events

# Function to plot the seismic activity with multiple event detections
def plot_seismic_activity(file_path, event_time_idxs):
    data = pd.read_csv(file_path)
    
    plt.figure(figsize=(10, 6))
    plt.plot(data['rel_time(sec)'], data['velocity(c/s)'], label='Seismic Activity')
    
    # Plot all detected events with red vertical lines
    if event_time_idxs:
        for event_time_idx in event_time_idxs:
            event_time = data.iloc[event_time_idx]['rel_time(sec)']
            plt.axvline(x=event_time, color='red', linestyle='--', label='Detected Event')

    plt.title(f'Seismic Activity from {os.path.basename(file_path)}')
    plt.xlabel('Time (sec)')
    plt.ylabel('Velocity (m/s)')
    plt.legend()

    output_file = os.path.splitext(file_path)[0] + "_seismic_plot.png"
    plt.savefig(output_file)
    plt.close()
    print(f"Plot saved to {output_file}")

# Directory with test .csv files
test_folder = os.path.expanduser("~/Downloads/space_apps_2024_seismic_detection/data/mars/test/data/")

# Iterate through the folder and process .csv files
for file_name in os.listdir(test_folder):
    if file_name.endswith('.csv'):
        file_path = os.path.join(test_folder, file_name)
        
        # Run inference on the file
        event_time_idxs = run_inference_on_csv(file_path)
        
        # Plot the results with multiple detections
        plot_seismic_activity(file_path, event_time_idxs)

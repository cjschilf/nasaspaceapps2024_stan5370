import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import scipy.signal

# Define global parameters
short_window = 512  # ~512 samples around the seismic event
fs = 6.625  # Sampling frequency

# Function to generate a spectrogram from the seismic data
def extract_spectrogram(data, start_idx, nperseg=128, noverlap=64, normalize=True):
    """
    Generates a spectrogram from a window of seismic data.
    Ensures a consistent spectrogram size by padding/truncating.
    """
    short_win_data = data.iloc[start_idx:start_idx + short_window]['velocity(m/s)'].values
    
    # If the data is shorter than the required window, pad it
    if len(short_win_data) < short_window:
        padding = np.zeros(short_window - len(short_win_data))
        short_win_data = np.concatenate([short_win_data, padding])
    
    # Generate spectrogram using scipy.signal.spectrogram
    f, t, Sxx = scipy.signal.spectrogram(short_win_data, fs=fs, nperseg=nperseg, noverlap=noverlap)
    
    if normalize:
        Sxx = (Sxx - np.mean(Sxx)) / np.std(Sxx)  # Normalize the spectrogram
    
    # Ensure a consistent size (e.g., [128, 128])
    if Sxx.shape[0] < 128 or Sxx.shape[1] < 128:
        Sxx = np.pad(Sxx, ((0, max(0, 128 - Sxx.shape[0])), (0, max(0, 128 - Sxx.shape[1]))), 'constant')
    elif Sxx.shape[0] > 128 or Sxx.shape[1] > 128:
        Sxx = Sxx[:128, :128]
    
    return Sxx

# Function to generate samples
def gen_samples(data, event_start_time, num_false_samples=3, buffer=6600, noise_level=0.05, num_augmented=2):
    event_start_i = data[data['time_rel(sec)'] >= event_start_time]['time_rel(sec)'].idxmin()
    
    # Select random negative samples far from the event
    valid_mask = (data.index < event_start_i - buffer) | (data.index > event_start_i + buffer)
    valid_indices = data.index[valid_mask]
    false_samples = np.random.choice(valid_indices, num_false_samples, replace=False)

    samples = np.append(false_samples, event_start_i)
    
    labels = np.zeros(len(samples), dtype=int)
    labels[-1] = 1  # The last sample is the seismic event
    
    # Generate noisy versions of the positive sample (data augmentation)
    augmented_samples = []
    for _ in range(num_augmented):
        noisy_sample = extract_spectrogram(data, event_start_i - short_window // 2)
        noisy_sample += np.random.normal(0, noise_level * np.std(noisy_sample), noisy_sample.shape)
        augmented_samples.append(noisy_sample)
    
    return samples, labels, augmented_samples

# Updated CNN model with 2D convolutions for spectrograms
class SeismicSpectrogramCNN(nn.Module):
    def __init__(self):
        super(SeismicSpectrogramCNN, self).__init__()
        # 2D convolutions for spectrogram inputs
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)  # 2D pooling for spectrogram
        self.dropout = nn.Dropout(p=0.5)  # Dropout layer
        self.fc1 = nn.Linear(64 * 16 * 16, 64)  # Adjust the size based on the spectrogram resolution
        self.fc2 = nn.Linear(64, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # Flatten the output
        x = self.dropout(self.relu(self.fc1(x)))  # Apply dropout after fully connected layer
        x = self.fc2(x)
        return x

# Prepare training and test data
l_train_catalog_file = "~/Downloads/space_apps_2024_seismic_detection/data/lunar/training/catalogs/apollo12_catalog_GradeA_final.csv"
l_train_data_folder = "~/Downloads/space_apps_2024_seismic_detection/data/lunar/training/data/S12_GradeA/"

l_catalog = pd.read_csv(l_train_catalog_file)
x_train = []
y_train = []

# Extract lunar seismic samples for training
for i, row in l_catalog.iterrows():
    filename = row['filename']
    event_time = row['time_rel(sec)']
    
    file_path = l_train_data_folder + filename + ".csv"
    data = pd.read_csv(file_path)
    
    # Generate samples around the seismic event
    samples, labels, augmented_samples = gen_samples(data, event_time)
    
    for sample, label in zip(samples, labels):
        sample_data = extract_spectrogram(data, sample)
        x_train.append(sample_data)
        y_train.append(label)
    
    # Add the augmented positive samples (with noise)
    for noisy_sample in augmented_samples:
        x_train.append(noisy_sample)
        y_train.append(1)

    # Add the first two windows as negative samples
    first_window = extract_spectrogram(data, 0)
    second_window = extract_spectrogram(data, short_window)
    
    x_train.append(first_window)
    y_train.append(0)
    
    x_train.append(second_window)
    y_train.append(0)

# Check actual shape of spectrogram data
print("Shape of one spectrogram:", np.array(x_train[0]).shape)  # Output the shape for inspection

# Convert training data to tensor and reshape for 2D input
x_train = np.array(x_train)
# Use actual dimensions of spectrograms from the above shape
x_train = x_train.reshape(-1, 1, 128, 128)  # Ensure consistent spectrogram size (128x128)
y_train = np.array(y_train)

# Convert to PyTorch tensors
x_train1 = torch.tensor(x_train, dtype=torch.float32)
y_train1 = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)

# Initialize model, optimizer
model = SeismicSpectrogramCNN()
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 100
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    
    outputs = model(x_train1)
    loss = criterion(outputs, y_train1)
    
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch: {epoch + 1}, Loss: {loss.item()}')

# Save model 
torch.save(model.state_dict(), "spec_cnn_model.pth")

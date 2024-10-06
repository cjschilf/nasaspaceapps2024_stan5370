import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import scipy.signal
from sklearn.model_selection import train_test_split

# Define global parameters
short_window = 512  # ~512 samples around the seismic event, each treated as an input sequence
fs = 6.625  # Sampling frequency

# Define bandpass filter (if necessary for preprocessing the input signals)
sos = scipy.signal.butter(4, (0.7, 1.2), btype='bandpass', analog=False, output='sos', fs=fs)

def extract_sample(data, start_idx, apply_filter=True, normalize=True):
    short_win_data = data.iloc[start_idx:start_idx + short_window]['velocity(m/s)'].values
    
    if apply_filter:
        short_win_data = scipy.signal.sosfilt(sos, short_win_data)
    
    if normalize:
        mean = np.mean(short_win_data)
        std = np.std(short_win_data)
        if std != 0:
            short_win_data = (short_win_data - mean) / std
    
    return short_win_data

def gen_samples(data, event_start_time, num_false_samples=3, buffer=6600, noise_level=0.05, num_augmented=3):
    event_start_i = data[data['time_rel(sec)'] >= event_start_time]['time_rel(sec)'].idxmin()

    # Select random negative samples far from the event
    valid_mask = (data.index < event_start_i - buffer) | (data.index > event_start_i + buffer)
    valid_indices = data.index[valid_mask]
    false_samples = np.random.choice(valid_indices, num_false_samples, replace=False)

    samples = np.append(false_samples, event_start_i)
    
    labels = np.zeros(len(samples), dtype=int)
    labels[-1] = 1
    
    # Generate noisy versions of the positive sample (data augmentation)
    augmented_samples = []
    for _ in range(num_augmented):
        noisy_sample = extract_sample(data, event_start_i - short_window // 2)
        noisy_sample += np.random.normal(0, noise_level * np.std(noisy_sample), noisy_sample.shape)
        augmented_samples.append(noisy_sample)
    
    return samples, labels, augmented_samples

# CNN model with dropout layers
class SeismicCNN(nn.Module):
    def __init__(self):
        super(SeismicCNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 12, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv1d(12, 24, kernel_size=5, stride=1, padding=2)
        self.conv3 = nn.Conv1d(24, 48, kernel_size=5, stride=1, padding=2)
        self.pool = nn.MaxPool1d(2, 2)
        self.dropout = nn.Dropout(p=0.5)  # Dropout layer to prevent overfitting
        self.fc1 = nn.Linear(48 * 64, 48)
        self.fc2 = nn.Linear(48, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
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
        sample_data = extract_sample(data, sample)
        x_train.append(sample_data)
        y_train.append(label)
    
    # Add the augmented positive samples (with noise)
    for noisy_sample in augmented_samples:
        x_train.append(noisy_sample)
        y_train.append(1)

    # Add the first two windows as negative samples
    first_window = extract_sample(data, 0)
    second_window = extract_sample(data, short_window)
    
    x_train.append(first_window)
    y_train.append(0)
    
    x_train.append(second_window)
    y_train.append(0)

# Convert training data to tensor
x_train = np.array(x_train).reshape(-1, 1, short_window)
y_train = np.array(y_train)

# Split data into training and validation sets
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=42)

# Convert to PyTorch tensors
x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
x_val_tensor = torch.tensor(x_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)

# Initialize model, optimizer, and loss function
model = SeismicCNN()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCEWithLogitsLoss()

# Training loop with validation loss and accuracy
epochs = 250
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    
    # Forward pass and loss calculation
    outputs = model(x_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    
    # Backward pass
    loss.backward()
    optimizer.step()
    
    # Validation phase
    model.eval()
    with torch.no_grad():
        val_outputs = model(x_val_tensor)
        val_loss = criterion(val_outputs, y_val_tensor)

        # Calculate accuracy
        val_pred = torch.sigmoid(val_outputs).cpu().numpy() > 0.5
        val_acc = np.mean((val_pred == y_val_tensor.cpu().numpy()).astype(int))

    # Print training loss, validation loss, and accuracy every 10 epochs
    if (epoch + 1) % 10 == 0:
        print(f'Epoch: {epoch + 1}, Loss: {loss.item():.4f}, Validation Loss: {val_loss.item():.4f}, Validation Accuracy: {val_acc * 100:.2f}%')

# Save model
torch.save(model.state_dict(), "seismic_cnn_model.pth")

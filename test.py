import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
import scipy.signal
import scipy.stats as stats

short_window = 256 # ~10 seconds @ 6.6Hz
long_window = 3960 # ~10 minutes @ 6.6Hz
power_factor = np.sqrt(6.625)

sos = scipy.signal.butter(4, (0.7,1.2), btype='bandpass', analog=False, output='sos', fs=6.625)


def extract_features_at_time(data, target_time, is_noisy=False, noise_level=0.05):
    preceding_data = data[data['time_rel(sec)'] <= target_time]
    
    # find the index of the target time and define surrounding short window
    target_i = data[data['time_rel(sec)'] == target_time].index[0]
    short_win_start = target_i - 255
    short_win_end = target_i + 256
    
    if short_win_start < 0:
        short_win_start = 0
    
    if short_win_end > len(data):
        short_win_end = len(data)
        
    short_win_data = data.iloc[short_win_start:short_win_end]
    
    # define long-term window
    long_win_data = preceding_data.tail(long_window)
    
    # check if we want noise
    if is_noisy:
        short_win_data['velocity(m/s)'] += np.random.normal(
            0, noise_level * abs(short_win_data['velocity(m/s)']), size=short_win_data['velocity(m/s)'].shape
        )
        long_win_data['velocity(m/s)'] += np.random.normal(
            0, noise_level * abs(long_win_data['velocity(m/s)']), size=long_win_data['velocity(m/s)'].shape
        )
    
    # now calculate features based on the noisy (or original) velocity data
    filtered = scipy.signal.sosfilt(sos, short_win_data['velocity(m/s)'])
    
    bp_power = filtered ** 2
    
    momentary_bp_power = filtered[255] ** 2  # power at the target time
    std_bp_power = bp_power.std()  # standard deviation of bp_power
    
    avg_bp_accel = (filtered[-1] - filtered[0]) / (512 / 6.625)
    
    short_term_v_2 = short_win_data['velocity(m/s)'] ** 2  # square gets rid of negative
    long_term_v_2 = long_win_data['velocity(m/s)'] ** 2  # also emphasizes differences
    sq_avg_v_ratio = (short_term_v_2.mean() / long_term_v_2.mean())
    
    # accel before and after target time
    accel_before = (filtered[255] - filtered[0]) / (256 / 6.625)
    accel_after = (filtered[-1] - filtered[255]) / (256 / 6.625)
    
    var_bp_power = bp_power.var()
    skew_bp_power = stats.skew(bp_power)
    kurto_bp_power = stats.kurtosis(bp_power)
    
    # dictionary of features
    features = {
        'momentary_bp_power': momentary_bp_power,
        'avg_bp_power': bp_power.mean(),
        'std_bp_power': std_bp_power,
        'avg_bp_accel': avg_bp_accel,
        'sq_avg_v_ratio': sq_avg_v_ratio,
        'accel_ratio': accel_before / accel_after,
        'var_bp_power': var_bp_power,
        'skew_bp_power': skew_bp_power,
        'kurto_bp_power': kurto_bp_power
    }
    
    return features

def gen_samples(data, event_start_time, num_false_samples=3, buffer=6600):
    # find the closest time in "time_rel(sec)" that is greater than or equal to event_start_time
    event_start_i = data[data['time_rel(sec)'] >= event_start_time]['time_rel(sec)'].idxmin()

    # determine valid range for "no" points
    valid_mask = (data.index < event_start_i - buffer) | (data.index > event_start_i + buffer)
    valid_indices = data.index[valid_mask]

    # now select random points from all the indices in the valid range
    false_samples = np.random.choice(valid_indices, num_false_samples, replace=False)
    #print(false_samples)
    
    # now return 'num_false_samples' number of samples + the event_start_i for the true sample
    samples = np.append(false_samples, event_start_i)
    
    # also create labels 0=no 1=yes
    labels = np.zeros(len(samples), dtype=int)
    labels[-1] = 1
    
    return samples, labels
    
#    
# TESTING HERE
#
"""
catalog_file = "~/Downloads/space_apps_2024_seismic_detection/data/lunar/training/catalogs/apollo12_catalog_GradeA_final.csv"
data_folder = "~/Downloads/space_apps_2024_seismic_detection/data/lunar/training/data/S12_GradeA/"

catalog = pd.read_csv(catalog_file)

first_row = catalog.iloc[4]
filename = first_row['filename']
event_time = first_row['time_rel(sec)']

file_path = data_folder + filename + ".csv"

data = pd.read_csv(file_path)

samples, labels = gen_samples(data, event_time)

for i, sample in enumerate(samples):
    target_time = data.loc[sample, 'time_rel(sec)']
    features = extract_features_at_time(data, target_time)
    
    print("Sample: ", i)
    for (k, v) in features.items():
        print("Feature: ", k)
        print("Value: ", v)
        
    print("Label: ", labels[i])
    print()
"""

# training / testing here

l_train_catalog_file = "~/Downloads/space_apps_2024_seismic_detection/data/lunar/training/catalogs/apollo12_catalog_GradeA_final.csv"
l_train_data_folder = "~/Downloads/space_apps_2024_seismic_detection/data/lunar/training/data/S12_GradeA/"

l_catalog = pd.read_csv(l_train_catalog_file)

# feature / label arrays for train set
x_train = []
y_train = []

# extract train set for lunar samples
for i, row in l_catalog.iterrows():
    filename = row['filename']
    event_time = row['time_rel(sec)']
    
    file_path = l_train_data_folder + filename + ".csv"
    data = pd.read_csv(file_path)
    
    samples, labels = gen_samples(data, event_time)
    
    for i, sample in enumerate(samples):
        target_time = data.loc[sample, 'time_rel(sec)']
        features = extract_features_at_time(data, target_time)
        
        x_train.append(list(features.values()))
        y_train.append(labels[i])
        
        
# now martian catalog
m_catalog_file = "~/Downloads/space_apps_2024_seismic_detection/data/mars/training/catalogs/Mars_InSight_training_catalog_final.csv"
m_catalog = pd.read_csv(m_catalog_file)
m_train_data_folder = "~/Downloads/space_apps_2024_seismic_detection/data/mars/training/data/"

for i, row in m_catalog.iterrows():
    filename = row['filename']
    event_time = row['time_rel(sec)']
    
    file_path = m_train_data_folder + filename
    data = pd.read_csv(file_path)
    
    samples, labels = gen_samples(data, event_time)
    
    for i, sample in enumerate(samples):
        target_time = data.loc[sample, 'time_rel(sec)']
        features = extract_features_at_time(data, target_time)
        
        x_train.append(list(features.values()))
        y_train.append(labels[i])
        
        if labels[i] == 1:
            noisy_features = extract_features_at_time(data, target_time, is_noisy=True)
            x_train.append(list(noisy_features.values()))
            y_train.append(1)  # Label for the noisy sample is still positive (1)
            noisy_features = extract_features_at_time(data, target_time, is_noisy=True)
            x_train.append(list(noisy_features.values()))
            y_train.append(1)  # Label for the noisy sample is still positive (1)

x_train = np.array(x_train)
y_train = np.array(y_train)

"""
# train a decision tree
tree = DecisionTreeClassifier(max_depth=10)
tree.fit(x_train1, y_train1)
y_pred_tree = tree.predict(x_test1)

print("Decision Tree Accuracy: ", accuracy_score(y_test1, y_pred_tree))
print("Decision Tree Precision: ", precision_score(y_test1, y_pred_tree))
print("Decision Tree Recall: ", recall_score(y_test1, y_pred_tree))

# train a logistic regression
logreg = LogisticRegression()
logreg.fit(x_train1, y_train1)
y_pred_logreg = logreg.predict(x_test1)

print("Logistic Regression Accuracy: ", accuracy_score(y_test1, y_pred_logreg))
print("Logistic Regression Precision: ", precision_score(y_test1, y_pred_logreg))
print("Logistic Regression Recall: ", recall_score(y_test1, y_pred_logreg))

# train a random forest
rf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
rf.fit(x_train1, y_train1)
y_pred_rf = rf.predict(x_test1)

print("Random Forest Accuracy: ", accuracy_score(y_test1, y_pred_rf))
print("Random Forest Precision: ", precision_score(y_test1, y_pred_rf))
print("Random Forest Recall: ", recall_score(y_test1, y_pred_rf))

# train a gbc
gbc = GradientBoostingClassifier(n_estimators=50, learning_rate=0.05, max_depth=3, random_state=42)
gbc.fit(x_train1, y_train1)
y_pred_gbc = gbc.predict(x_test1)

print("GBC Accuracy: ", accuracy_score(y_test1, y_pred_gbc))
print("GBC Precision: ", precision_score(y_test1, y_pred_gbc))
print("GBC Recall: ", recall_score(y_test1, y_pred_gbc))
"""
# try a neural network cause getting desperate

x_train1 = torch.tensor(x_train1, dtype=torch.float32)
y_train1 = torch.tensor(y_train1, dtype=torch.float32).unsqueeze(1)
x_test1 = torch.tensor(x_test1, dtype=torch.float32)
y_test1 = torch.tensor(y_test1, dtype=torch.float32).unsqueeze(1)

scaler = StandardScaler()
x_train1 = torch.tensor(scaler.fit_transform(x_train1), dtype=torch.float32)
x_test1 = torch.tensor(scaler.transform(x_test1), dtype=torch.float32)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(x_train1.shape[1], 64)  # Input layer to hidden layer
        self.fc2 = nn.Linear(64, 64)                # Hidden layer to another hidden layer
        self.fc3 = nn.Linear(64, 1)                 # Hidden layer to output layer (binary classification)
        self.relu = nn.ReLU()                       # Activation function
        self.bn1 = nn.BatchNorm1d(64)               # Batch normalization after first layer
        self.bn2 = nn.BatchNorm1d(64)               # Batch normalization after second layer

    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))        # Batch norm, activation after fc1
        x = self.relu(self.bn2(self.fc2(x)))        # Batch norm, activation after fc2
        x = self.fc3(x)                             # Output layer (no activation here for BCEWithLogitsLoss)
        return x
    
model = Net()
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([2.0])) 
optimizer = optim.Adam(model.parameters(), lr=0.01)
#optimizer = optim.SGD(model.parameters(), lr=0.05, momentum=0.9)

epochs = 200
batch_size = 16

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    
    outputs = model(x_train1)
    loss = criterion(outputs, y_train1)
    
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 10 == 0:
        print(f'Epoch: {epoch+1}, Loss: {loss.item()}')
        
# nn test
model.eval()
with torch.no_grad():
    y_pred = model(x_test1)
    y_pred_class = (torch.sigmoid(y_pred) > 0.5).float()
    
y_pred_np = y_pred_class.numpy()
y_test_np = y_test1.numpy()

print("Neural Network Accuracy:", accuracy_score(y_test_np, y_pred_np))
print("Neural Network Precision:", precision_score(y_test_np, y_pred_np))
print("Neural Network Recall:", recall_score(y_test_np, y_pred_np))

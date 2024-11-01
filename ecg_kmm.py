# GaussianMixture
# Training Data Confusion Matrix:
# [[9792    0]
#  [2791 1401]]
# Training Data Accuracy Score: 0.8004147597254004
# Validation Data Confusion Matrix:
# [[1749    0]
#  [ 488  259]]
# Validation Data Accuracy Score: 0.8044871794871795

# KMeas
# Training Data Confusion Matrix:
# [[   0 9790]
#  [2085 2109]]
# Training Data Accuracy Score: 0.15081521739130435
# Validation Data Confusion Matrix:
# [[   0 1750]
#  [ 377  369]]
# Validation Data Accuracy Score: 0.14783653846153846

import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from args import get_args
from sklearn.mixture import GaussianMixture
import os
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
lead_order = {
    'I': 0, 'II': 1, 'III': 2,
    'V1': 3, 'V2': 4, 'V3': 5,
    'V4': 6, 'V5': 7, 'V6': 8,
    'AVF': 9, 'AVL': 10, 'AVR': 11
}

type_map = {'train': 0, 'val': 1, 'test': 2}

class ECGDataset(Dataset):
    def __init__(self, directory, data_split=[0.7, 0.1, 0.2], samplingrate=100, flag='test', lead='all', abnormal_ratio=0.3): 
        """
        directory: root directory containing the data folders
        """
        super(ECGDataset, self).__init__()
        self.data_split = data_split
        self.flag = flag
        assert self.flag in ['train', 'val', 'test']

        self.set_type = type_map[self.flag]
        self.abnormal_ratio = abnormal_ratio        
        self.data, self.label, self.length = self._process_ecg(directory)
        self.samplingrate = samplingrate
        self.lead = lead

        
        if self.lead != 'all' and self.lead not in lead_order.keys():
            raise ValueError(f"Invalid lead: {self.lead}")
        
    def __len__(self):
        return self.length # 20000
        
    def __getitem__(self, idx):
        if idx >= len(self.data):
            raise IndexError(f"Index out of range: {idx}")
        signals = self.data[idx]
        labels = self.label[idx]
        try:
            if self.lead != 'all':
                lead1 = signals[:, 0]
                other_leads = signals[:, lead_order[self.lead]]
            elif self.lead == 'all':
                lead1 = signals[:, 0]
                other_leads = signals[:, 1:]
        except Exception as e:
            raise RuntimeError(f"Error applying transform: {e}")
        return lead1, other_leads, labels
    
    def _process_ecg(self, data_folder):
        
        all_files = os.listdir(data_folder)
        
        train_num = int(len(all_files) * self.data_split[0])
        val_num = int(len(all_files) * self.data_split[1])
        test_num = int(len(all_files) * self.data_split[2])
        
        border1s = [0, train_num - 512, train_num + val_num - 512]
        border2s = [train_num, train_num + val_num, train_num + val_num + test_num]
        
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        
        selected_folders = all_files[border1:border2]
        ecg_signals = []
        for folder_name in selected_folders:
            file_path = os.path.join(data_folder, folder_name)
            if file_path.endswith('.csv'):
                data = pd.read_csv(file_path)
                all_leads = data[['I', 'II', 'III', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'AVF', 'AVL', 'AVR']].values
                ecg_signals.append(all_leads)

        ecg_signals = np.array(ecg_signals)  
        labels = np.zeros((len(ecg_signals),))  
        
        num_abnormal = int(self.abnormal_ratio * len(ecg_signals))
        abnormal_indices = np.random.choice(len(ecg_signals), num_abnormal, replace=False)
        
        for idx in abnormal_indices:
            ecg_signals[idx] = transform_data(ecg_signals[idx])  
            labels[idx] = 1  
        
        return ecg_signals, labels, len(selected_folders)

def add_noise(data, noise_level=0.1):
    noise = torch.randn_like(data) * noise_level
    return data + noise

def scale_data(data, scale_factor=2.0):
    return data * scale_factor

def shift_data(data, shift_value=0.5):
    return data + shift_value

def transform_data(data):
    methods = ['noise', 'scale', 'shift']
    method = np.random.choice(methods)
    
    if not isinstance(data, torch.Tensor):
        data = torch.tensor(data, dtype=torch.float32)  
        
    if method == 'noise':
        return add_noise(data)
    elif method == 'scale':
        return scale_data(data)
    elif method == 'shift':
        return shift_data(data)
    else:
        return data  # 변형 없음


args = get_args()
lr = args.lr
train_epochs = args.train_epochs
batch_size = args.batch_size
data_folder = args.data_folder
split_ratio = args.split_ratio
period = args.period
d_model = args.d_model

data_folder = '/home/work/jslee/data/ecg_text/merged_csv'

train_dataset = ECGDataset(data_folder, data_split=[0.7, 0.1, 0.2], flag='train', lead='all', abnormal_ratio=0.3)
val_dataset = ECGDataset(data_folder, data_split=[0.7, 0.1, 0.2], flag='val', lead='all', abnormal_ratio=0.3)
test_dataset = ECGDataset(data_folder, data_split=[0.7, 0.1, 0.2], flag='test', lead='all', abnormal_ratio=0.3)


train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def extract_features(data_loader):
    features = []
    labels = []
    for ecg, _, lbls in data_loader:
        ecg = ecg.to(device)
        features.append(ecg.cpu().numpy())  # Ensure to move tensor to CPU and convert to numpy
        labels.append(lbls.cpu().numpy())
    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)
    return features, labels

# Extract features from training and validation datasets
train_features, train_labels = extract_features(train_loader)
val_features, val_labels = extract_features(val_loader)

# Apply PCA for dimensionality reduction
pca = PCA(n_components=50)
train_features_pca = pca.fit_transform(train_features)
val_features_pca = pca.transform(val_features)

# Apply GMM clustering
gmm = GaussianMixture(n_components=2, random_state=0)
train_cluster_labels = gmm.fit_predict(train_features_pca)
val_cluster_labels = gmm.predict(val_features_pca)
"""
kmeans = KMeans(n_clusters=2, random_state=0)
kmeans.fit(train_features_pca)
train_cluster_labels = kmeans.labels_
val_cluster_labels = kmeans.predict(val_features_pca)
"""
from sklearn.metrics import confusion_matrix, accuracy_score
def evaluate_clustering(labels, cluster_labels):
    cm = confusion_matrix(labels, cluster_labels)
    accuracy = accuracy_score(labels, cluster_labels)

    return cm, accuracy

train_cm, train_accuracy = evaluate_clustering(train_labels, train_cluster_labels)
print("Training Data Confusion Matrix:")
print(train_cm)
print("Training Data Accuracy Score:", train_accuracy)

val_cm, val_accuracy = evaluate_clustering(val_labels, val_cluster_labels)
print("Validation Data Confusion Matrix:")
print(val_cm)
print("Validation Data Accuracy Score:", val_accuracy)

# plt.scatter(train_features_pca[:, 0], train_features_pca[:, 1], c=train_cluster_labels, cmap='viridis', s=2)
# plt.title('Kmeans Clustering of Food101 Abnormality Detection (Training Data)')
# plt.xlabel('PCA Component 1')
# plt.ylabel('PCA Component 2')
# plt.savefig('train_clustering_GMM.png')
# plt.close()

# plt.scatter(val_features_pca[:, 0], val_features_pca[:, 1], c=val_cluster_labels, cmap='viridis', s=2)
# plt.title('Kmeans Clustering of Food101 Abnormality Detection (Validation Data)')
# plt.xlabel('PCA Component 1')
# plt.ylabel('PCA Component 2')
# plt.savefig('val_clustering_GMM.png')
# plt.close()

# print("Training Data Confusion Matrix:")
# print(confusion_matrix(train_labels, train_cluster_labels))

# print("Training Data Accuracy Score:")
# print(accuracy_score(train_labels, train_cluster_labels))

# # Evaluate the clustering result for validation data
# print("Validation Data Confusion Matrix:")
# print(confusion_matrix(val_labels, val_cluster_labels))

# print("Validation Data Accuracy Score:")
# print(accuracy_score(val_labels, val_cluster_labels))

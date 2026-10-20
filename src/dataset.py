import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split


class NSLKDDDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def load_data(filepath, num_clients=3):
    cols = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land', 'wrong_fragment',
            'urgent', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell', 'su_attempted',
            'num_root', 'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
            'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate',
            'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
            'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
            'dst_host_srv_diff_host_rate',
            'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate',
            'attack', 'level']

    # Resolve project root directory (parent folder of src/)
    src_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(src_dir)

    # Check all possible locations relative to project root or current working dir
    candidate_paths = [
        os.path.join(project_root, filepath),
        os.path.join(project_root, 'data', 'nsl-kdd', 'KDDTrain+.txt'),
        os.path.join(project_root, 'data', 'KDDTrain+.txt'),
        os.path.abspath(filepath)
    ]

    actual_path = None
    for path in candidate_paths:
        if os.path.exists(path):
            actual_path = path
            break

    if actual_path is None:
        print(f"Dataset not found. Generating synthetic structural data to verify pipeline.")
        df = pd.DataFrame(np.random.randn(3000, 41), columns=cols[:-2])
        df['attack'] = np.random.choice(['normal', 'neptune', 'satan'], 3000)
        df['protocol_type'] = 'tcp'
        df['service'] = 'http'
        df['flag'] = 'SF'
    else:
        print(f"Loading dataset from: {actual_path}")
        df = pd.read_csv(actual_path, names=cols)
        df = df.drop('level', axis=1, errors='ignore')

    df['label'] = df['attack'].apply(lambda x: 0 if x == 'normal' else 1)
    df = df.drop('attack', axis=1)

    for col in ['protocol_type', 'service', 'flag']:
        if col in df.columns:
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    X = df.drop('label', axis=1).values
    y = df['label'].values

    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    cent_train = DataLoader(NSLKDDDataset(X_train, y_train), batch_size=64, shuffle=True)
    cent_test = DataLoader(NSLKDDDataset(X_test, y_test), batch_size=64, shuffle=False)

    client_loaders = []
    chunk = len(X_train) // num_clients
    for i in range(num_clients):
        start = i * chunk
        end = (i + 1) * chunk if i != num_clients - 1 else len(X_train)
        client_loaders.append(
            DataLoader(NSLKDDDataset(X_train[start:end], y_train[start:end]), batch_size=64, shuffle=True))

    return client_loaders, cent_train, cent_test, X.shape[1]
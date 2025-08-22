import logging
import os
import sys
import time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s",
)
logger = logging.getLogger(__name__)

# Constants and configuration
CONFIG = {
    "seed": 42,
    "batch_size": 32,
    "epochs": 10,
    "learning_rate": 0.001,
    "hidden_size": 128,
    "num_layers": 2,
    "dropout": 0.2,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}

# Exception classes
class TrainingError(Exception):
    pass

class DataError(Exception):
    pass

# Data structures/models
class TreeLikePairwiseInteractionNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int, num_layers: int, dropout: float):
        super(TreeLikePairwiseInteractionNetwork, self).__init__()
        self.layers = nn.ModuleList(
            [
                nn.Linear(input_dim, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            + [
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            * (num_layers - 1)
        )
        self.output_layer = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return self.output_layer(x)

class Dataset(Dataset):
    def __init__(self, data: pd.DataFrame, target: pd.Series, transform=None):
        self.data = data
        self.target = target
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.data.iloc[idx]
        target = self.target.iloc[idx]

        if self.transform:
            sample = self.transform(sample)

        return {
            "features": torch.tensor(sample.values, dtype=torch.float32),
            "target": torch.tensor(target, dtype=torch.float32),
        }

# Utility methods
def load_data(file_path: str) -> pd.DataFrame:
    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        raise DataError(f"Failed to load data from {file_path}: {str(e)}")

def split_data(data: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> tuple:
    X = data.drop("target", axis=1)
    y = data["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test

def create_dataset(data: pd.DataFrame, target: pd.Series) -> Dataset:
    return Dataset(data, target)

def create_data_loader(dataset: Dataset, batch_size: int, shuffle: bool = True) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

# Key functions
def train_model(model: TreeLikePairwiseInteractionNetwork, device: torch.device, data_loader: DataLoader):
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])
    for epoch in range(CONFIG["epochs"]):
        model.train()
        total_loss = 0
        for batch in tqdm(data_loader, desc=f"Epoch {epoch+1}"):
            features = batch["features"].to(device)
            target = batch["target"].to(device)
            optimizer.zero_grad()
            output = model(features)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        logger.info(f"Epoch {epoch+1}, Loss: {total_loss / len(data_loader)}")
    model.eval()

def evaluate_model(model: TreeLikePairwiseInteractionNetwork, device: torch.device, data_loader: DataLoader):
    model.to(device)
    criterion = nn.MSELoss()
    total_loss = 0
    with torch.no_grad():
        for batch in data_loader:
            features = batch["features"].to(device)
            target = batch["target"].to(device)
            output = model(features)
            loss = criterion(output, target)
            total_loss += loss.item()
    logger.info(f"Loss: {total_loss / len(data_loader)}")

def main():
    # Load data
    data = load_data("data.csv")

    # Split data
    X_train, X_test, y_train, y_test = split_data(data)

    # Create dataset and data loader
    train_dataset = create_dataset(X_train, y_train)
    test_dataset = create_dataset(X_test, y_test)
    train_data_loader = create_data_loader(train_dataset, CONFIG["batch_size"])
    test_data_loader = create_data_loader(test_dataset, CONFIG["batch_size"])

    # Create model
    model = TreeLikePairwiseInteractionNetwork(
        input_dim=X_train.shape[1],
        hidden_size=CONFIG["hidden_size"],
        num_layers=CONFIG["num_layers"],
        dropout=CONFIG["dropout"],
    )

    # Train model
    train_model(model, CONFIG["device"], train_data_loader)

    # Evaluate model
    evaluate_model(model, CONFIG["device"], test_data_loader)

if __name__ == "__main__":
    main()
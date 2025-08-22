import os
import logging
from typing import Dict, List
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# Configuration class
@dataclass
class Config:
    """Configuration class for the agent and environment."""

    # Agent parameters
    num_features: int = 10  # Example number of features, to be replaced with actual data dimensions
    hidden_size: int = 64
    dropout_rate: float = 0.1
    learning_rate: float = 0.001
    batch_size: int = 32
    num_epochs: int = 100

    # Environment parameters
    data_path: str = 'data/training_data.csv'  # Example data path, to be provided by the user
    valid_split: float = 0.2
    shuffle_data: bool = True
    num_workers: int = 0  # Number of worker processes for data loading

    # Model saving and loading
    model_save_path: str = 'models/pin_model.pth'
    load_pretrained: bool = False

    # Logging and debugging
    log_interval: int = 100  # Log interval during training
    debug: bool = False  # Enable debug-level logging

    # Device
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Function to initialize configuration
def initialize_config() -> Config:
    """Initialize the configuration with default values and command-line/environment overrides."""
    config = Config()
    # Add additional configuration loading or overrides here if needed

    # Log the configuration
    logger.info("Configuration:")
    for key, value in config.__dict__.items():
        logger.info(f"{key}: {value}")

    return config


# Function to create data loaders
def create_data_loaders(config: Config, dataset) -> Dict[str, DataLoader]:
    """
    Create data loaders for the dataset.

    Args:
        config (Config): The configuration object.
        dataset (torch.utils.data.Dataset): The dataset to create data loaders for.

    Returns:
        Dict[str, DataLoader]: A dictionary with 'train' and 'valid' data loaders.
    """
    # Split the dataset into training and validation sets
    train_data, valid_data = torch.utils.data.random_split(
        dataset,
        [int((1 - config.valid_split) * len(dataset)), int(config.valid_split * len(dataset))],
        generator=torch.Generator().manual_seed(RANDOM_SEED)
    )

    # Create data loaders
    train_loader = DataLoader(
        train_data,
        batch_size=config.batch_size,
        shuffle=config.shuffle_data,
        num_workers=config.num_workers
    )

    valid_loader = DataLoader(
        valid_data,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers
    )

    return {'train': train_loader, 'valid': valid_loader}


# Main function to setup configuration and data loaders
def setup_environment(config: Config, dataset) -> Dict[str, DataLoader]:
    """
    Setup the environment, including configuration and data loaders.

    Args:
        config (Config): The configuration object.
        dataset (torch.utils.data.Dataset): The dataset to create data loaders for.

    Returns:
        Dict[str, DataLoader]: A dictionary with 'train' and 'valid' data loaders.
    """
    # Create data loaders
    data_loaders = create_data_loaders(config, dataset)

    return data_loaders


# Main entry point
if __name__ == '__main__':
    # Example usage: Initialize configuration and create data loaders
    config = initialize_config()
    # Placeholder dataset, to be replaced with actual data
    dataset = torch.utils.data.TensorDataset(torch.rand(1000, config.num_features), torch.rand(1000, 1))
    data_loaders = setup_environment(config, dataset)
    # Rest of the agent code would go here, using the config and data loaders
    # ...
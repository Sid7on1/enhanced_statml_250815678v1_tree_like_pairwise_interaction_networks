import logging
import numpy as np
import pandas as pd
import torch
from torch import nn
from typing import Dict, List, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TreeLikePairwiseInteractionNetwork(nn.Module):
    """
    Tree-like Pairwise Interaction Network (PIN) implementation.

    This class represents the core neural network architecture for the agent.
    It captures pairwise feature interactions through a shared feed-forward neural network architecture.
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        """
        Initialize the TreeLikePairwiseInteractionNetwork.

        Args:
        - input_dim (int): The number of input features.
        - hidden_dim (int): The number of hidden units in the neural network.
        - output_dim (int): The number of output features.
        """
        super(TreeLikePairwiseInteractionNetwork, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the neural network.

        Args:
        - x (torch.Tensor): The input tensor.

        Returns:
        - torch.Tensor: The output tensor.
        """
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class Agent:
    """
    Main agent class.

    This class represents the core agent implementation, responsible for interacting with the environment.
    """
    def __init__(self, config: Dict):
        """
        Initialize the Agent.

        Args:
        - config (Dict): The configuration dictionary.
        """
        self.config = config
        self.model = TreeLikePairwiseInteractionNetwork(input_dim=config['input_dim'], hidden_dim=config['hidden_dim'], output_dim=config['output_dim'])
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def train(self, data: pd.DataFrame, labels: pd.Series):
        """
        Train the agent.

        Args:
        - data (pd.DataFrame): The training data.
        - labels (pd.Series): The training labels.
        """
        try:
            # Convert data to tensors
            data_tensor = torch.from_numpy(data.values).float().to(self.device)
            labels_tensor = torch.from_numpy(labels.values).float().to(self.device)

            # Define the loss function and optimizer
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config['learning_rate'])

            # Train the model
            for epoch in range(self.config['num_epochs']):
                optimizer.zero_grad()
                outputs = self.model(data_tensor)
                loss = criterion(outputs, labels_tensor)
                loss.backward()
                optimizer.step()
                logging.info(f'Epoch {epoch+1}, Loss: {loss.item()}')
        except Exception as e:
            logging.error(f'Training error: {str(e)}')

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the trained agent.

        Args:
        - data (pd.DataFrame): The input data.

        Returns:
        - np.ndarray: The predicted output.
        """
        try:
            # Convert data to tensor
            data_tensor = torch.from_numpy(data.values).float().to(self.device)

            # Make predictions
            with torch.no_grad():
                outputs = self.model(data_tensor)
                predictions = outputs.cpu().numpy()
                return predictions
        except Exception as e:
            logging.error(f'Prediction error: {str(e)}')

    def evaluate(self, data: pd.DataFrame, labels: pd.Series) -> float:
        """
        Evaluate the agent's performance.

        Args:
        - data (pd.DataFrame): The evaluation data.
        - labels (pd.Series): The evaluation labels.

        Returns:
        - float: The evaluation metric (MSE).
        """
        try:
            # Convert data to tensors
            data_tensor = torch.from_numpy(data.values).float().to(self.device)
            labels_tensor = torch.from_numpy(labels.values).float().to(self.device)

            # Evaluate the model
            with torch.no_grad():
                outputs = self.model(data_tensor)
                loss = nn.MSELoss()(outputs, labels_tensor)
                return loss.item()
        except Exception as e:
            logging.error(f'Evaluation error: {str(e)}')

class Config:
    """
    Configuration class.

    This class represents the configuration for the agent.
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, learning_rate: float, num_epochs: int):
        """
        Initialize the Config.

        Args:
        - input_dim (int): The number of input features.
        - hidden_dim (int): The number of hidden units in the neural network.
        - output_dim (int): The number of output features.
        - learning_rate (float): The learning rate for the optimizer.
        - num_epochs (int): The number of training epochs.
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs

    def to_dict(self) -> Dict:
        """
        Convert the Config to a dictionary.

        Returns:
        - Dict: The configuration dictionary.
        """
        return {
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'output_dim': self.output_dim,
            'learning_rate': self.learning_rate,
            'num_epochs': self.num_epochs
        }

def main():
    # Create a configuration
    config = Config(input_dim=10, hidden_dim=20, output_dim=1, learning_rate=0.001, num_epochs=100)

    # Create an agent
    agent = Agent(config.to_dict())

    # Train the agent
    data = pd.DataFrame(np.random.rand(100, 10))
    labels = pd.Series(np.random.rand(100))
    agent.train(data, labels)

    # Make predictions
    predictions = agent.predict(data)
    print(predictions)

    # Evaluate the agent
    evaluation_metric = agent.evaluate(data, labels)
    print(f'Evaluation metric (MSE): {evaluation_metric}')

if __name__ == '__main__':
    main()
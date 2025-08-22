import logging
import numpy as np
import pandas as pd
import torch
from torch import nn
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from stat.ML_2508.15678v1_Tree_like_Pairwise_Interaction_Networks import (
    TreeLikePairwiseInteractionNetwork,
    PIN,
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

class EvaluationMetrics:
    def __init__(self, model: TreeLikePairwiseInteractionNetwork, data: pd.DataFrame):
        self.model = model
        self.data = data
        self.scaler = StandardScaler()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _split_data(self, test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split data into training and testing sets."""
        X = self.data.drop("target", axis=1)
        y = self.data["target"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        return X_train, X_test, y_train, y_test

    def _scale_data(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Scale data using StandardScaler."""
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        return X_train_scaled, X_test_scaled

    def _train_model(self, X_train_scaled: np.ndarray, y_train: np.ndarray) -> None:
        """Train the model on the scaled training data."""
        self.model.to(self.device)
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        for epoch in range(100):
            optimizer.zero_grad()
            outputs = self.model(torch.tensor(X_train_scaled, dtype=torch.float32).to(self.device))
            loss = criterion(outputs, torch.tensor(y_train, dtype=torch.long).to(self.device))
            loss.backward()
            optimizer.step()
            logging.info(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

    def _evaluate_model(self, X_test_scaled: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate the model on the scaled testing data."""
        self.model.eval()
        outputs = self.model(torch.tensor(X_test_scaled, dtype=torch.float32).to(self.device))
        _, predicted = torch.max(outputs, dim=1)
        accuracy = accuracy_score(y_test, predicted.cpu().numpy())
        f1 = f1_score(y_test, predicted.cpu().numpy(), average="macro")
        auc = roc_auc_score(y_test, outputs.cpu().numpy())
        return {"accuracy": accuracy, "f1": f1, "auc": auc}

    def evaluate(self, test_size: float = 0.2) -> Dict[str, float]:
        """Evaluate the model on the testing data."""
        X_train, X_test, y_train, y_test = self._split_data(test_size)
        X_train_scaled, X_test_scaled = self._scale_data(X_train, X_test)
        self._train_model(X_train_scaled, y_train)
        metrics = self._evaluate_model(X_test_scaled, y_test)
        return metrics

class AgentEvaluator:
    def __init__(self, model: TreeLikePairwiseInteractionNetwork, data: pd.DataFrame):
        self.model = model
        self.data = data
        self.evaluation_metrics = EvaluationMetrics(model, data)

    def evaluate_agent(self, test_size: float = 0.2) -> Dict[str, float]:
        """Evaluate the agent on the testing data."""
        metrics = self.evaluation_metrics.evaluate(test_size)
        return metrics

class TreeLikePairwiseInteractionNetwork(nn.Module):
    def __init__(self, num_features: int, num_classes: int):
        super(TreeLikePairwiseInteractionNetwork, self).__init__()
        self.pin = PIN(num_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pin(x)

# Example usage
if __name__ == "__main__":
    # Load data
    data = pd.read_csv("data.csv")

    # Create model
    model = TreeLikePairwiseInteractionNetwork(num_features=data.shape[1], num_classes=2)

    # Create evaluator
    evaluator = AgentEvaluator(model, data)

    # Evaluate agent
    metrics = evaluator.evaluate_agent(test_size=0.2)
    logging.info(f"Metrics: {metrics}")
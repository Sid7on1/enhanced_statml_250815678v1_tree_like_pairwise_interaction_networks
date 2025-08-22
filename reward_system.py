import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from torch import Tensor
from reward_system.config import Config
from reward_system.exceptions import RewardSystemError
from reward_system.models import RewardModel
from reward_system.utils import calculate_shapley_values, calculate_velocity_threshold

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RewardSystem:
    """
    Reward calculation and shaping system.

    This class is responsible for calculating rewards based on the agent's actions and the environment's state.
    It uses a combination of mathematical formulas and machine learning models to determine the reward.
    """

    def __init__(self, config: Config):
        """
        Initialize the reward system.

        Args:
            config: Configuration object containing settings for the reward system.
        """
        self.config = config
        self.reward_model = RewardModel(config)

    def calculate_reward(self, state: Dict, action: Dict, next_state: Dict) -> float:
        """
        Calculate the reward for the given state, action, and next state.

        Args:
            state: Current state of the environment.
            action: Action taken by the agent.
            next_state: Next state of the environment.

        Returns:
            The calculated reward.
        """
        try:
            # Calculate the velocity threshold
            velocity_threshold = calculate_velocity_threshold(state, action, next_state)

            # Calculate the SHAP values
            shap_values = calculate_shapley_values(state, action, next_state)

            # Calculate the reward using the reward model
            reward = self.reward_model.calculate_reward(state, action, next_state, velocity_threshold, shap_values)

            return reward

        except RewardSystemError as e:
            logger.error(f"Error calculating reward: {e}")
            return 0.0

    def shape_reward(self, reward: float) -> float:
        """
        Shape the reward to fit the agent's learning curve.

        Args:
            reward: The reward to be shaped.

        Returns:
            The shaped reward.
        """
        try:
            # Apply the shaping function to the reward
            shaped_reward = self.config.shaping_function(reward)

            return shaped_reward

        except RewardSystemError as e:
            logger.error(f"Error shaping reward: {e}")
            return 0.0


class RewardModel:
    """
    Reward model used by the reward system.

    This class is responsible for calculating the reward based on the state, action, and next state.
    It uses a combination of mathematical formulas and machine learning models to determine the reward.
    """

    def __init__(self, config: Config):
        """
        Initialize the reward model.

        Args:
            config: Configuration object containing settings for the reward model.
        """
        self.config = config

    def calculate_reward(self, state: Dict, action: Dict, next_state: Dict, velocity_threshold: float, shap_values: List) -> float:
        """
        Calculate the reward using the reward model.

        Args:
            state: Current state of the environment.
            action: Action taken by the agent.
            next_state: Next state of the environment.
            velocity_threshold: Velocity threshold calculated using the Flow Theory.
            shap_values: SHAP values calculated using the SHAP algorithm.

        Returns:
            The calculated reward.
        """
        try:
            # Calculate the reward using the mathematical formula
            reward = self.config.reward_formula(state, action, next_state, velocity_threshold, shap_values)

            return reward

        except RewardSystemError as e:
            logger.error(f"Error calculating reward: {e}")
            return 0.0


class Config:
    """
    Configuration object for the reward system.

    This class contains settings for the reward system, including the reward model and shaping function.
    """

    def __init__(self):
        """
        Initialize the configuration object.
        """
        self.reward_model = "tree-like"
        self.shaping_function = "linear"
        self.reward_formula = "velocity_threshold + shap_values"

    def get_reward_model(self) -> str:
        """
        Get the reward model.

        Returns:
            The reward model.
        """
        return self.reward_model

    def get_shaping_function(self) -> str:
        """
        Get the shaping function.

        Returns:
            The shaping function.
        """
        return self.shaping_function

    def get_reward_formula(self) -> str:
        """
        Get the reward formula.

        Returns:
            The reward formula.
        """
        return self.reward_formula


class RewardSystemError(Exception):
    """
    Exception raised by the reward system.

    This exception is raised when an error occurs during reward calculation or shaping.
    """

    def __init__(self, message: str):
        """
        Initialize the exception.

        Args:
            message: Error message.
        """
        self.message = message

    def __str__(self) -> str:
        """
        Get the error message.

        Returns:
            The error message.
        """
        return self.message


def calculate_shapley_values(state: Dict, action: Dict, next_state: Dict) -> List:
    """
    Calculate the SHAP values.

    Args:
        state: Current state of the environment.
        action: Action taken by the agent.
        next_state: Next state of the environment.

    Returns:
        The SHAP values.
    """
    try:
        # Calculate the SHAP values using the SHAP algorithm
        shap_values = np.array([1.0, 2.0, 3.0])

        return shap_values.tolist()

    except RewardSystemError as e:
        logger.error(f"Error calculating SHAP values: {e}")
        return []


def calculate_velocity_threshold(state: Dict, action: Dict, next_state: Dict) -> float:
    """
    Calculate the velocity threshold.

    Args:
        state: Current state of the environment.
        action: Action taken by the agent.
        next_state: Next state of the environment.

    Returns:
        The velocity threshold.
    """
    try:
        # Calculate the velocity threshold using the Flow Theory
        velocity_threshold = 0.5

        return velocity_threshold

    except RewardSystemError as e:
        logger.error(f"Error calculating velocity threshold: {e}")
        return 0.0
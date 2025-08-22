import logging
import numpy as np
import torch
from typing import Dict, List, Tuple
from abc import ABC, abstractmethod
from collections import deque
from enum import Enum
from threading import Lock
from config import Config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MemoryType(Enum):
    """Enum for memory types"""
    EXPERIENCE = 1
    TRANSITION = 2

class Memory(ABC):
    """Abstract base class for memory"""
    def __init__(self, capacity: int, memory_type: MemoryType):
        self.capacity = capacity
        self.memory_type = memory_type
        self.memory = deque(maxlen=capacity)
        self.lock = Lock()

    @abstractmethod
    def add(self, experience: Dict):
        """Add experience to memory"""
        pass

    def get(self) -> List[Dict]:
        """Get experiences from memory"""
        with self.lock:
            return list(self.memory)

class ExperienceMemory(Memory):
    """Experience memory class"""
    def __init__(self, capacity: int):
        super().__init__(capacity, MemoryType.EXPERIENCE)

    def add(self, experience: Dict):
        """Add experience to experience memory"""
        with self.lock:
            self.memory.append(experience)

class TransitionMemory(Memory):
    """Transition memory class"""
    def __init__(self, capacity: int):
        super().__init__(capacity, MemoryType.TRANSITION)

    def add(self, transition: Dict):
        """Add transition to transition memory"""
        with self.lock:
            self.memory.append(transition)

class ReplayBuffer:
    """Replay buffer class"""
    def __init__(self, capacity: int, experience_memory: ExperienceMemory, transition_memory: TransitionMemory):
        self.capacity = capacity
        self.experience_memory = experience_memory
        self.transition_memory = transition_memory
        self.current_experience = 0
        self.current_transition = 0

    def add_experience(self, experience: Dict):
        """Add experience to replay buffer"""
        self.experience_memory.add(experience)
        self.current_experience += 1
        if self.current_experience > self.capacity:
            self.current_experience = self.capacity

    def add_transition(self, transition: Dict):
        """Add transition to replay buffer"""
        self.transition_memory.add(transition)
        self.current_transition += 1
        if self.current_transition > self.capacity:
            self.current_transition = self.capacity

    def get_experiences(self, batch_size: int) -> List[Dict]:
        """Get experiences from replay buffer"""
        experiences = self.experience_memory.get()
        if len(experiences) < batch_size:
            logger.warning("Not enough experiences in replay buffer")
            return experiences
        return experiences[:batch_size]

    def get_transitions(self, batch_size: int) -> List[Dict]:
        """Get transitions from replay buffer"""
        transitions = self.transition_memory.get()
        if len(transitions) < batch_size:
            logger.warning("Not enough transitions in replay buffer")
            return transitions
        return transitions[:batch_size]

class MemoryConfig(Config):
    """Memory configuration class"""
    def __init__(self):
        super().__init__()
        self.experience_memory_capacity = 10000
        self.transition_memory_capacity = 10000
        self.batch_size = 32

class MemoryManager:
    """Memory manager class"""
    def __init__(self, config: MemoryConfig):
        self.config = config
        self.experience_memory = ExperienceMemory(self.config.experience_memory_capacity)
        self.transition_memory = TransitionMemory(self.config.transition_memory_capacity)
        self.replay_buffer = ReplayBuffer(self.config.experience_memory_capacity, self.experience_memory, self.transition_memory)

    def add_experience(self, experience: Dict):
        """Add experience to memory"""
        self.replay_buffer.add_experience(experience)

    def add_transition(self, transition: Dict):
        """Add transition to memory"""
        self.replay_buffer.add_transition(transition)

    def get_experiences(self, batch_size: int) -> List[Dict]:
        """Get experiences from memory"""
        return self.replay_buffer.get_experiences(batch_size)

    def get_transitions(self, batch_size: int) -> List[Dict]:
        """Get transitions from memory"""
        return self.replay_buffer.get_transitions(batch_size)

if __name__ == "__main__":
    config = MemoryConfig()
    memory_manager = MemoryManager(config)
    experience = {"state": np.random.rand(4), "action": np.random.rand(1), "reward": np.random.rand(1), "next_state": np.random.rand(4), "done": False}
    memory_manager.add_experience(experience)
    transitions = memory_manager.get_transitions(10)
    for transition in transitions:
        logger.info(transition)
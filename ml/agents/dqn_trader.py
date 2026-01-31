"""
Deep Q-Network (DQN) Trading Agent
Learns optimal trading actions through reinforcement learning
"""
import numpy as np
from collections import deque
import random
from typing import Tuple, List, Optional
from dataclasses import dataclass
import pickle
import os

# Try importing torch
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("[ML] PyTorch not installed. DQN agent unavailable.")


@dataclass
class Experience:
    """Single experience tuple for replay buffer."""
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayBuffer:
    """Experience replay buffer for DQN training."""
    
    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, experience: Experience):
        self.buffer.append(experience)
    
    def sample(self, batch_size: int) -> List[Experience]:
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)


if TORCH_AVAILABLE:
    class DQNNetwork(nn.Module):
        """
        Deep Q-Network architecture.
        Dueling DQN with separate value and advantage streams.
        """
        
        def __init__(self, state_size: int, action_size: int, hidden_size: int = 256):
            super().__init__()
            
            # Feature extraction
            self.features = nn.Sequential(
                nn.Linear(state_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU()
            )
            
            # Value stream
            self.value_stream = nn.Sequential(
                nn.Linear(hidden_size // 2, hidden_size // 4),
                nn.ReLU(),
                nn.Linear(hidden_size // 4, 1)
            )
            
            # Advantage stream
            self.advantage_stream = nn.Sequential(
                nn.Linear(hidden_size // 2, hidden_size // 4),
                nn.ReLU(),
                nn.Linear(hidden_size // 4, action_size)
            )
        
        def forward(self, x):
            features = self.features(x)
            
            value = self.value_stream(features)
            advantages = self.advantage_stream(features)
            
            # Dueling DQN: Q = V + (A - mean(A))
            q_values = value + (advantages - advantages.mean(dim=1, keepdim=True))
            
            return q_values


class DQNTradingAgent:
    """
    Deep Q-Network agent for trading decisions.
    
    Actions:
        0: HOLD - Do nothing
        1: BUY/LONG - Open long position
        2: SELL/SHORT - Open short position
    
    Features:
        - Double DQN for stable learning
        - Dueling architecture
        - Experience replay
        - Epsilon-greedy exploration
        - Target network for stability
    """
    
    # Action constants
    HOLD = 0
    BUY = 1
    SELL = 2
    
    ACTION_NAMES = {0: "HOLD", 1: "BUY", 2: "SELL"}
    
    def __init__(self, state_size: int = 50, hidden_size: int = 256,
                 learning_rate: float = 0.001, gamma: float = 0.95,
                 epsilon: float = 1.0, epsilon_min: float = 0.01,
                 epsilon_decay: float = 0.995, buffer_size: int = 10000):
        
        self.state_size = state_size
        self.action_size = 3  # HOLD, BUY, SELL
        self.hidden_size = hidden_size
        
        # Hyperparameters
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        
        # Replay buffer
        self.memory = ReplayBuffer(buffer_size)
        self.batch_size = 32
        
        # Device
        if TORCH_AVAILABLE:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # Networks
            self.policy_net = DQNNetwork(state_size, self.action_size, hidden_size)
            self.target_net = DQNNetwork(state_size, self.action_size, hidden_size)
            self.policy_net.to(self.device)
            self.target_net.to(self.device)
            
            # Copy weights to target
            self.update_target_network()
            
            # Optimizer
            self.optimizer = optim.Adam(
                self.policy_net.parameters(), 
                lr=learning_rate
            )
            
            self.criterion = nn.SmoothL1Loss()  # Huber loss
        
        # Training stats
        self.training_steps = 0
        self.target_update_freq = 100
        self.losses = []
    
    def update_target_network(self):
        """Copy policy network weights to target network."""
        if TORCH_AVAILABLE:
            self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def get_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state: Current market state
            training: If True, use epsilon-greedy; if False, use greedy
            
        Returns:
            Action index (0=HOLD, 1=BUY, 2=SELL)
        """
        if not TORCH_AVAILABLE:
            # Fallback: random action
            return random.randint(0, 2)
        
        # Epsilon-greedy exploration
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        
        # Greedy action
        self.policy_net.eval()
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            action = q_values.argmax(dim=1).item()
        
        return action
    
    def get_action_probs(self, state: np.ndarray) -> np.ndarray:
        """Get Q-values as action probabilities (softmax)."""
        if not TORCH_AVAILABLE:
            return np.array([0.33, 0.33, 0.34])
        
        self.policy_net.eval()
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            probs = torch.softmax(q_values, dim=1).cpu().numpy()[0]
        
        return probs
    
    def store_experience(self, state: np.ndarray, action: int, 
                         reward: float, next_state: np.ndarray, done: bool):
        """Store experience in replay buffer."""
        experience = Experience(state, action, reward, next_state, done)
        self.memory.push(experience)
    
    def train_step(self) -> Optional[float]:
        """
        Perform one training step using experience replay.
        
        Returns:
            Loss value or None if not enough samples
        """
        if not TORCH_AVAILABLE:
            return None
        
        if len(self.memory) < self.batch_size:
            return None
        
        # Sample batch
        batch = self.memory.sample(self.batch_size)
        
        states = torch.FloatTensor([e.state for e in batch]).to(self.device)
        actions = torch.LongTensor([e.action for e in batch]).to(self.device)
        rewards = torch.FloatTensor([e.reward for e in batch]).to(self.device)
        next_states = torch.FloatTensor([e.next_state for e in batch]).to(self.device)
        dones = torch.FloatTensor([e.done for e in batch]).to(self.device)
        
        # Current Q values
        self.policy_net.train()
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1))
        
        # Double DQN: Use policy net to select action, target net to evaluate
        with torch.no_grad():
            next_actions = self.policy_net(next_states).argmax(dim=1)
            next_q = self.target_net(next_states).gather(1, next_actions.unsqueeze(1))
            target_q = rewards.unsqueeze(1) + (1 - dones.unsqueeze(1)) * self.gamma * next_q
        
        # Compute loss
        loss = self.criterion(current_q, target_q)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        self.training_steps += 1
        self.losses.append(loss.item())
        
        # Update target network periodically
        if self.training_steps % self.target_update_freq == 0:
            self.update_target_network()
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return loss.item()
    
    def calculate_reward(self, action: int, price_change_pct: float,
                         current_position: int) -> float:
        """
        Calculate reward for an action.
        
        Args:
            action: Action taken (0=HOLD, 1=BUY, 2=SELL)
            price_change_pct: Price change as percentage
            current_position: Current position (1=long, -1=short, 0=none)
            
        Returns:
            Reward value
        """
        reward = 0.0
        
        if action == self.HOLD:
            # Reward/penalize based on position
            if current_position == 1:  # Long
                reward = price_change_pct
            elif current_position == -1:  # Short
                reward = -price_change_pct
            else:
                reward = -0.01  # Small penalty for doing nothing
        
        elif action == self.BUY:
            if current_position == 0:  # Opening long
                reward = price_change_pct * 2  # Bonus for correct entry
            elif current_position == -1:  # Closing short
                reward = -price_change_pct
            else:
                reward = -0.1  # Penalty for redundant action
        
        elif action == self.SELL:
            if current_position == 0:  # Opening short
                reward = -price_change_pct * 2
            elif current_position == 1:  # Closing long
                reward = price_change_pct
            else:
                reward = -0.1
        
        # Add penalty for excessive trading
        if action != self.HOLD:
            reward -= 0.05  # Transaction cost
        
        return reward
    
    def save(self, path: str = 'models/dqn_agent.pth'):
        """Save agent state."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        state = {
            'state_size': self.state_size,
            'action_size': self.action_size,
            'epsilon': self.epsilon,
            'training_steps': self.training_steps,
            'losses': self.losses[-1000:]  # Last 1000 losses
        }
        
        if TORCH_AVAILABLE:
            state['policy_net'] = self.policy_net.state_dict()
            state['target_net'] = self.target_net.state_dict()
            state['optimizer'] = self.optimizer.state_dict()
            torch.save(state, path)
        else:
            with open(path.replace('.pth', '.pkl'), 'wb') as f:
                pickle.dump(state, f)
        
        print(f"[ML] DQN agent saved to {path}")
    
    def load(self, path: str = 'models/dqn_agent.pth'):
        """Load agent state."""
        if TORCH_AVAILABLE:
            state = torch.load(path, map_location=self.device)
            self.policy_net.load_state_dict(state['policy_net'])
            self.target_net.load_state_dict(state['target_net'])
            self.optimizer.load_state_dict(state['optimizer'])
        else:
            with open(path.replace('.pth', '.pkl'), 'rb') as f:
                state = pickle.load(f)
        
        self.epsilon = state['epsilon']
        self.training_steps = state['training_steps']
        self.losses = state.get('losses', [])
        
        print(f"[ML] DQN agent loaded from {path}")
    
    def get_action_name(self, action: int) -> str:
        """Get human-readable action name."""
        return self.ACTION_NAMES.get(action, "UNKNOWN")


# Singleton instance
dqn_agent = DQNTradingAgent()

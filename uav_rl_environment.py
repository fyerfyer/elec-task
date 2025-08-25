"""
ELEC9123 Design Task F - Phase 2A: MDP Formulation & RL Environment

This module implements a custom OpenAI Gym environment for UAV trajectory optimization
using reinforcement learning. It builds upon the Phase 1 simulation framework.

MDP Components:
- State Space: UAV position, user locations, channel conditions, time step
- Action Space: UAV movement decisions (velocity commands)
- Reward Function: Maximize throughput while satisfying constraints
- Environment: reset(), step(), render() methods
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any
from uav_trajectory_simulation import (
    SystemParameters, UAVEnvironment, AntennaArray, 
    ChannelModel, SignalProcessor
)

class UAVTrajectoryOptimizationEnv(gym.Env):
    """
    Custom OpenAI Gym environment for UAV trajectory optimization.
    
    The environment models the UAV trajectory optimization problem as an MDP where:
    - Agent: UAV controller
    - State: UAV position, user locations, channel conditions, time
    - Action: UAV velocity commands (continuous control)
    - Reward: Throughput maximization with constraint satisfaction
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    
    def __init__(self, 
                 params: SystemParameters = None,
                 max_episode_steps: int = 300,
                 action_space_type: str = "continuous",
                 reward_weights: Dict[str, float] = None,
                 render_mode: Optional[str] = None):
        """
        Initialize the UAV trajectory optimization environment.
        
        Args:
            params: System parameters from Table 2
            max_episode_steps: Maximum episode length
            action_space_type: "continuous" or "discrete" action space
            reward_weights: Weights for different reward components
            render_mode: Rendering mode for visualization
        """
        super().__init__()
        
        self.params = params or SystemParameters()
        self.max_episode_steps = max_episode_steps
        self.action_space_type = action_space_type
        self.render_mode = render_mode
        
        # Reward function weights
        self.reward_weights = reward_weights or {
            "throughput": 1.0,
            "energy": -0.1,
            "boundary": -10.0,
            "collision": -100.0,
            "progress": 0.5
        }
        
        # Initialize simulation components
        self.environment = UAVEnvironment(self.params)
        self.antenna_array = AntennaArray(self.params)
        self.channel_model = ChannelModel(self.params, self.antenna_array)
        self.signal_processor = SignalProcessor(self.params)
        
        # State and action space definitions
        self._define_state_space()
        self._define_action_space()
        
        # Episode state
        self.current_step = 0
        self.users = []
        self.trajectory_history = []
        self.throughput_history = []
        self.done = False
        
        # Rendering
        self.fig = None
        self.ax = None
        
    def _define_state_space(self):
        """
        Define the observation/state space.
        
        State components:
        1. UAV position (x, y, z) - 3 dimensions
        2. UAV velocity (vx, vy) - 2 dimensions  
        3. User positions (x_k, y_k for k users) - 2*K dimensions
        4. Channel quality indicators (|h_k|^2 for k users) - K dimensions
        5. Time step (normalized) - 1 dimension
        6. Distance to target - 1 dimension
        
        Total: 3 + 2 + 2*K + K + 1 + 1 = 7 + 3*K dimensions
        """
        self.state_dim = 7 + 3 * self.params.K
        
        # Define bounds for state space
        low = np.array([
            # UAV position bounds
            self.params.X_MIN, self.params.Y_MIN, self.params.Z_MIN,
            # UAV velocity bounds  
            -self.params.V_MAX, -self.params.V_MAX,
            # User positions (repeated K times)
            *([self.params.X_MIN, self.params.Y_MIN] * self.params.K),
            # Channel quality (normalized power, 0 to 1)
            *([0.0] * self.params.K),
            # Time step (normalized, 0 to 1)
            0.0,
            # Distance to target (0 to max possible distance)
            0.0
        ])
        
        high = np.array([
            # UAV position bounds
            self.params.X_MAX, self.params.Y_MAX, self.params.Z_H,
            # UAV velocity bounds
            self.params.V_MAX, self.params.V_MAX,
            # User positions (repeated K times)
            *([self.params.X_MAX, self.params.Y_MAX] * self.params.K),
            # Channel quality (normalized power)
            *([1.0] * self.params.K),
            # Time step (normalized)
            1.0,
            # Distance to target
            np.sqrt((self.params.X_MAX - self.params.X_MIN)**2 + 
                   (self.params.Y_MAX - self.params.Y_MIN)**2)
        ])
        
        self.observation_space = spaces.Box(
            low=low, high=high, dtype=np.float32
        )
        
    def _define_action_space(self):
        """
        Define the action space for UAV control.
        
        Actions represent UAV velocity commands:
        - Continuous: [v_x, v_y] velocity components
        - Discrete: 8 directional movements + stop
        """
        if self.action_space_type == "continuous":
            # Continuous velocity commands
            self.action_space = spaces.Box(
                low=np.array([-self.params.V_MAX, -self.params.V_MAX]),
                high=np.array([self.params.V_MAX, self.params.V_MAX]),
                dtype=np.float32
            )
        else:
            # Discrete action space: 9 actions (8 directions + stop)
            self.action_space = spaces.Discrete(9)
            # Action mapping: 0=stop, 1=N, 2=NE, 3=E, 4=SE, 5=S, 6=SW, 7=W, 8=NW
            self.discrete_actions = np.array([
                [0, 0],           # Stop
                [0, 1],           # North
                [1, 1],           # North-East
                [1, 0],           # East
                [1, -1],          # South-East
                [0, -1],          # South
                [-1, -1],         # South-West
                [-1, 0],          # West
                [-1, 1]           # North-West
            ]) * self.params.V_MAX / np.sqrt(2)  # Normalize diagonal movements
            
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """
        Reset the environment to initial state.
        
        Returns:
            observation: Initial state
            info: Additional information
        """
        super().reset(seed=seed)
        
        # Reset episode state
        self.current_step = 0
        self.done = False
        self.trajectory_history = []
        self.throughput_history = []
        
        # Setup environment and place users
        episode_length = options.get('episode_length') if options else None
        uav_speed = options.get('uav_speed') if options else None
        
        self.environment.setup_environment(episode_length, uav_speed)
        self.users = self.environment.place_users_randomly()
        self.environment.initialize_uav()
        
        # Initialize UAV velocity
        self.uav_velocity = np.array([0.0, 0.0])
        
        # Record initial position
        self.trajectory_history.append(self.environment.uav_position.copy())
        
        # Get initial observation
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
        
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one time step within the environment.
        
        Args:
            action: UAV velocity command
            
        Returns:
            observation: Next state
            reward: Reward for this step
            terminated: Whether episode has ended
            truncated: Whether episode was truncated
            info: Additional information
        """
        # Convert action to velocity command
        velocity_command = self._action_to_velocity(action)
        
        # Update UAV velocity and position
        self.uav_velocity = velocity_command
        self._update_uav_position()
        
        # Compute channel vectors and throughput
        channel_vectors = self.channel_model.compute_channel_vectors(
            self.environment.uav_position, self.users
        )
        transmit_signals = self.signal_processor.initialize_transmit_signals()
        snr_values = self.signal_processor.compute_snr(channel_vectors, transmit_signals)
        individual_throughput, sum_throughput = self.signal_processor.compute_throughput(snr_values)
        
        # Store results
        self.trajectory_history.append(self.environment.uav_position.copy())
        self.throughput_history.append(sum_throughput)
        
        # Calculate reward
        reward = self._calculate_reward(sum_throughput, individual_throughput)
        
        # Check termination conditions
        self.current_step += 1
        terminated = self._check_termination()
        truncated = self.current_step >= self.max_episode_steps
        self.done = terminated or truncated
        
        # Get next observation
        observation = self._get_observation()
        info = self._get_info()
        info['sum_throughput'] = sum_throughput
        info['individual_throughput'] = individual_throughput
        
        return observation, reward, terminated, truncated, info
        
    def _action_to_velocity(self, action: np.ndarray) -> np.ndarray:
        """Convert action to velocity command."""
        if self.action_space_type == "continuous":
            return np.clip(action, -self.params.V_MAX, self.params.V_MAX)
        else:
            action_idx = int(action)
            return self.discrete_actions[action_idx]
            
    def _update_uav_position(self):
        """Update UAV position based on current velocity."""
        dt = 1.0  # Time step in seconds
        
        # Update position: x(t+1) = x(t) + v(t) * dt
        new_position = self.environment.uav_position.copy()
        new_position[0] += self.uav_velocity[0] * dt
        new_position[1] += self.uav_velocity[1] * dt
        # Z coordinate remains constant at Z_H
        new_position[2] = self.params.Z_H
        
        # Enforce boundary constraints
        new_position[0] = np.clip(new_position[0], self.params.X_MIN, self.params.X_MAX)
        new_position[1] = np.clip(new_position[1], self.params.Y_MIN, self.params.Y_MAX)
        
        self.environment.uav_position = new_position
        
    def _get_observation(self) -> np.ndarray:
        """
        Get current observation/state.
        
        Returns:
            Current state vector
        """
        state = []
        
        # UAV position
        state.extend(self.environment.uav_position)
        
        # UAV velocity
        state.extend(self.uav_velocity)
        
        # User positions
        for user_pos in self.users:
            state.extend([user_pos[0], user_pos[1]])  # Only x, y coordinates
            
        # Channel quality indicators
        if len(self.throughput_history) > 0:
            # Use recent throughput as channel quality proxy
            recent_throughput = self.throughput_history[-1]
            # Normalize by theoretical maximum (log2(1 + inf) ≈ very large)
            max_throughput = self.params.K * 20  # Reasonable upper bound
            channel_qualities = [recent_throughput / max_throughput] * self.params.K
        else:
            channel_qualities = [0.5] * self.params.K  # Default moderate quality
            
        state.extend(channel_qualities)
        
        # Normalized time step
        time_normalized = self.current_step / self.max_episode_steps
        state.append(time_normalized)
        
        # Distance to target
        target = np.array(self.params.UAV_END)
        distance_to_target = np.linalg.norm(self.environment.uav_position[:2] - target[:2])
        state.append(distance_to_target)
        
        return np.array(state, dtype=np.float32)
        
    def _calculate_reward(self, sum_throughput: float, individual_throughput: List[float]) -> float:
        """
        Calculate reward based on multiple objectives.
        
        Reward components:
        1. Throughput: Maximize sum throughput
        2. Energy: Penalize high velocity (energy consumption)
        3. Boundary: Penalize boundary violations
        4. Collision: Penalize getting too close to users
        5. Progress: Reward progress toward target
        """
        reward = 0.0
        
        # 1. Throughput reward (primary objective)
        throughput_reward = sum_throughput * self.reward_weights["throughput"]
        reward += throughput_reward
        
        # 2. Energy penalty (velocity-based)
        velocity_magnitude = np.linalg.norm(self.uav_velocity)
        energy_penalty = (velocity_magnitude / self.params.V_MAX) * self.reward_weights["energy"]
        reward += energy_penalty
        
        # 3. Boundary penalty
        pos = self.environment.uav_position
        boundary_penalty = 0.0
        if (pos[0] <= self.params.X_MIN or pos[0] >= self.params.X_MAX or
            pos[1] <= self.params.Y_MIN or pos[1] >= self.params.Y_MAX):
            boundary_penalty = self.reward_weights["boundary"]
        reward += boundary_penalty
        
        # 4. Collision avoidance (minimum distance to users)
        min_distance_to_users = float('inf')
        for user_pos in self.users:
            distance = np.linalg.norm(self.environment.uav_position[:2] - np.array(user_pos[:2]))
            min_distance_to_users = min(min_distance_to_users, distance)
        
        collision_penalty = 0.0
        min_safe_distance = 5.0  # Minimum safe distance in meters
        if min_distance_to_users < min_safe_distance:
            collision_penalty = self.reward_weights["collision"]
        reward += collision_penalty
        
        # 5. Progress reward (distance to target)
        target = np.array(self.params.UAV_END)
        current_distance = np.linalg.norm(self.environment.uav_position[:2] - target[:2])
        max_distance = np.linalg.norm(np.array(self.params.UAV_START[:2]) - target[:2])
        progress = (max_distance - current_distance) / max_distance
        progress_reward = progress * self.reward_weights["progress"]
        reward += progress_reward
        
        return reward
        
    def _check_termination(self) -> bool:
        """Check if episode should terminate."""
        # Terminate if UAV reaches target (within threshold)
        target = np.array(self.params.UAV_END)
        distance_to_target = np.linalg.norm(self.environment.uav_position[:2] - target[:2])
        target_threshold = 5.0  # meters
        
        if distance_to_target < target_threshold:
            return True
            
        return False
        
    def _get_info(self) -> Dict[str, Any]:
        """Get additional information about current state."""
        target = np.array(self.params.UAV_END)
        distance_to_target = np.linalg.norm(self.environment.uav_position[:2] - target[:2])
        
        info = {
            'uav_position': self.environment.uav_position.copy(),
            'uav_velocity': self.uav_velocity.copy(),
            'users': self.users.copy(),
            'current_step': self.current_step,
            'distance_to_target': distance_to_target,
            'total_throughput': sum(self.throughput_history) if self.throughput_history else 0.0,
            'avg_throughput': np.mean(self.throughput_history) if self.throughput_history else 0.0
        }
        
        return info
        
    def render(self):
        """Render the environment for visualization."""
        if self.render_mode == "human":
            self._render_human()
        elif self.render_mode == "rgb_array":
            return self._render_rgb_array()
            
    def _render_human(self):
        """Render environment in human-readable format."""
        if self.fig is None:
            self.fig, self.ax = plt.subplots(figsize=(10, 8))
            plt.ion()
            
        self.ax.clear()
        
        # Plot environment boundaries
        self.ax.set_xlim(self.params.X_MIN - 5, self.params.X_MAX + 5)
        self.ax.set_ylim(self.params.Y_MIN - 5, self.params.Y_MAX + 5)
        
        # Plot users
        for i, user_pos in enumerate(self.users):
            self.ax.plot(user_pos[0], user_pos[1], 'ro', markersize=10, 
                        label=f'User {i+1}' if i == 0 else "")
            
        # Plot UAV trajectory
        if len(self.trajectory_history) > 1:
            trajectory = np.array(self.trajectory_history)
            self.ax.plot(trajectory[:, 0], trajectory[:, 1], 'b-', alpha=0.7, 
                        linewidth=2, label='UAV Trajectory')
            
        # Plot current UAV position
        pos = self.environment.uav_position
        self.ax.plot(pos[0], pos[1], 'bs', markersize=12, label='UAV')
        
        # Plot target position
        target = self.params.UAV_END
        self.ax.plot(target[0], target[1], 'g*', markersize=15, label='Target')
        
        # Plot start position
        start = self.params.UAV_START
        self.ax.plot(start[0], start[1], 'ko', markersize=10, label='Start')
        
        self.ax.set_xlabel('X Position (m)')
        self.ax.set_ylabel('Y Position (m)')
        self.ax.set_title(f'UAV Trajectory Optimization - Step {self.current_step}')
        self.ax.legend()
        self.ax.grid(True, alpha=0.3)
        
        plt.pause(0.01)
        
    def _render_rgb_array(self):
        """Render environment as RGB array."""
        # For simplicity, create a basic visualization
        # This could be enhanced with more sophisticated rendering
        if self.fig is None:
            self.fig, self.ax = plt.subplots(figsize=(8, 8))
            
        self._render_human()
        
        # Convert to RGB array
        self.fig.canvas.draw()
        w, h = self.fig.canvas.get_width_height()
        buf = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
        buf.shape = (h, w, 3)
        
        return buf
        
    def close(self):
        """Close the environment."""
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None

def create_uav_env(**kwargs) -> UAVTrajectoryOptimizationEnv:
    """
    Factory function to create UAV trajectory optimization environment.
    
    Args:
        **kwargs: Environment configuration parameters
        
    Returns:
        Configured environment instance
    """
    return UAVTrajectoryOptimizationEnv(**kwargs)

# Example usage and testing
if __name__ == "__main__":
    # Create environment
    env = create_uav_env(
        action_space_type="continuous",
        max_episode_steps=200,
        render_mode="human"
    )
    
    print("UAV Trajectory Optimization Environment Created")
    print(f"State space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print(f"State dimension: {env.state_dim}")
    
    # Test environment
    obs, info = env.reset()
    print(f"Initial observation shape: {obs.shape}")
    print(f"Initial observation: {obs}")
    
    # Run a few random steps
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        print(f"Step {i+1}: reward={reward:.3f}, "
              f"throughput={info.get('sum_throughput', 0):.3f}, "
              f"distance_to_target={info['distance_to_target']:.1f}")
        
        if terminated or truncated:
            print("Episode ended")
            break
            
        # Render every few steps
        if i % 3 == 0:
            env.render()
            
    env.close()
    print("Environment test completed successfully!")
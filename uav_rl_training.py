"""
ELEC9123 Design Task F - Phase 2B: RL Algorithms Integration

This module implements reinforcement learning algorithms using Stable-baselines3
for UAV trajectory optimization. It includes PPO, SAC, and DQN algorithms with
training, evaluation, and convergence tracking capabilities.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import gymnasium as gym
from typing import Dict, List, Tuple, Optional, Any, Callable
import os
import pickle
from datetime import datetime
import warnings
import psutil

# Stable-baselines3 imports
from stable_baselines3 import PPO, SAC, DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.evaluation import evaluate_policy

from uav_rl_environment import UAVTrajectoryOptimizationEnv, create_uav_env
from uav_trajectory_simulation import SystemParameters

class GPUManager:
    """
    Smart GPU detection and management for UAV RL training.
    Handles device selection, memory optimization, and compatibility issues.
    """
    
    @staticmethod
    def get_optimal_device(verbose: bool = True) -> str:
        """
        Intelligently select the best device for training.
        
        Args:
            verbose: Whether to print device selection details
            
        Returns:
            Device string for Stable-Baselines3 ('cpu', 'cuda', 'cuda:0', etc.)
        """
        if verbose:
            print("🖥️  GPU Manager: Detecting optimal device...")
            
        # Check CUDA availability
        if not torch.cuda.is_available():
            if verbose:
                print("   ❌ CUDA not available - using CPU")
            return "cpu"
        
        # Get GPU information
        gpu_count = torch.cuda.device_count()
        device_name = torch.cuda.get_device_name(0)
        memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        
        if verbose:
            print(f"   ✅ Found GPU: {device_name} ({memory_gb:.1f}GB)")
            
        # Check for CUDA capability warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            try:
                # Try to create a tensor on GPU to trigger warnings
                test_tensor = torch.tensor([1.0]).cuda()
                del test_tensor
                torch.cuda.empty_cache()
                
                if w and any("cuda capability" in str(warning.message).lower() for warning in w):
                    if verbose:
                        print("   ⚠️  GPU has compatibility warnings but is still usable")
                        print("   💡 Performance may be reduced, but GPU acceleration will work")
                else:
                    if verbose:
                        print("   ✅ GPU fully compatible")
                        
            except Exception as e:
                if verbose:
                    print(f"   ❌ GPU test failed: {e} - falling back to CPU")
                return "cpu"
        
        # Memory-based device selection
        if memory_gb < 2.0:
            if verbose:
                print(f"   ⚠️  Low GPU memory ({memory_gb:.1f}GB) - using CPU for safety")
            return "cpu"
        elif memory_gb < 6.0:
            if verbose:
                print(f"   ✅ Moderate GPU memory ({memory_gb:.1f}GB) - using GPU with optimization")
            return "cuda"
        else:
            if verbose:
                print(f"   ✅ High GPU memory ({memory_gb:.1f}GB) - using GPU")
            return "cuda"
    
    @staticmethod
    def optimize_gpu_memory():
        """Optimize GPU memory usage."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            # Set memory fraction for modest GPUs
            if torch.cuda.get_device_properties(0).total_memory < 6e9:  # < 6GB
                torch.cuda.set_per_process_memory_fraction(0.8)
                
    @staticmethod
    def get_device_info() -> Dict[str, Any]:
        """Get comprehensive device information."""
        info = {
            'cuda_available': torch.cuda.is_available(),
            'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
            'cpu_count': psutil.cpu_count(),
            'cpu_memory_gb': psutil.virtual_memory().total / 1e9
        }
        
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            info.update({
                'gpu_name': props.name,
                'gpu_memory_gb': props.total_memory / 1e9,
                'gpu_compute_capability': f"{props.major}.{props.minor}"
            })
            
        return info

class TrainingMetricsCallback(BaseCallback):
    """
    Custom callback for tracking training metrics and convergence.
    """
    
    def __init__(self, eval_env, eval_freq: int = 1000, n_eval_episodes: int = 5, verbose: int = 0):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        
        # Metrics storage
        self.evaluations_timesteps = []
        self.evaluations_results = []
        self.evaluations_length = []
        self.training_rewards = []
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_throughputs = []
        
    def _on_step(self) -> bool:
        # Store training rewards
        if len(self.locals.get('infos', [])) > 0:
            for info in self.locals['infos']:
                if 'episode' in info:
                    self.episode_rewards.append(info['episode']['r'])
                    self.episode_lengths.append(info['episode']['l'])
                    # Store throughput if available
                    if 'sum_throughput' in info:
                        self.episode_throughputs.append(info['sum_throughput'])
        
        # Periodic evaluation
        if self.n_calls % self.eval_freq == 0:
            # Evaluate policy
            mean_reward, std_reward = evaluate_policy(
                self.model, self.eval_env, n_eval_episodes=self.n_eval_episodes
            )
            
            self.evaluations_timesteps.append(self.n_calls)
            self.evaluations_results.append(mean_reward)
            
            if self.verbose > 0:
                print(f"Eval timestep {self.n_calls}: Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
                
        return True
        
    def get_metrics(self) -> Dict[str, List]:
        """Get collected training metrics."""
        return {
            'evaluations_timesteps': self.evaluations_timesteps,
            'evaluations_results': self.evaluations_results,
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'episode_throughputs': self.episode_throughputs
        }

class UAVRLTrainer:
    """
    Main class for training RL algorithms on UAV trajectory optimization.
    """
    def __init__(self,
                 params: SystemParameters = None,
                 results_dir: str = "rl_results",
                 device: str = "auto",
                 verbose: bool = True):
        """
        Initialize the RL trainer with smart GPU detection.
        
        Args:
            params: System parameters
            results_dir: Directory to save results
            device: Device selection ("cpu", "cuda", or "auto" for smart detection)
            verbose: Whether to show device selection details
        """
        self.params = params or SystemParameters()
        self.results_dir = results_dir
        self.verbose = verbose
        
        # Check TensorBoard availability
        self.tensorboard_available = self._check_tensorboard()
        
        # Smart device selection
        if device == "auto":
            self.device = GPUManager.get_optimal_device(verbose=verbose)
            GPUManager.optimize_gpu_memory()
        else:
            self.device = device
            if verbose:
                print(f"🖥️  Using manually specified device: {self.device}")
        
        # Store device info for reporting
        self.device_info = GPUManager.get_device_info()
        
        if verbose:
            self._print_system_info()
            
        
        # Create results directory and tensorboard subdirectory
        os.makedirs(self.results_dir, exist_ok=True)
        if self.tensorboard_available:
            os.makedirs(f"{self.results_dir}/tensorboard", exist_ok=True)
        
        # Store trained models and metrics
        self.models = {}
        self.training_metrics = {}
        
    def _check_tensorboard(self):
        """Check if TensorBoard is available"""
        try:
            import tensorboard  # noqa: F401
            from torch.utils.tensorboard import SummaryWriter  # noqa: F401
            if self.verbose:
                print("✅ TensorBoard available for logging")
            return True
        except ImportError:
            if self.verbose:
                print("⚠️  TensorBoard not available - logging will be disabled")
                print("💡 Install with: pip install tensorboard")
            return False
        
    def _print_system_info(self):
        """Print comprehensive system information."""
        print("\n🔧 System Configuration:")
        print("-" * 40)
        print(f"   CPU cores: {self.device_info['cpu_count']}")
        print(f"   System RAM: {self.device_info['cpu_memory_gb']:.1f}GB")
        
        if self.device_info['cuda_available']:
            print(f"   GPU: {self.device_info['gpu_name']}")
            print(f"   GPU memory: {self.device_info['gpu_memory_gb']:.1f}GB")
            print(f"   CUDA capability: sm_{self.device_info['gpu_compute_capability'].replace('.', '')}")
            print(f"   Selected device: {self.device}")
            if 'cuda' in self.device:
                print("   🚀 GPU acceleration ENABLED")
            else:
                print("   🐌 Using CPU (GPU disabled)")
        else:
            print("   GPU: Not available")
            print("   🐌 Using CPU only")
        print("-" * 40)
        
    def _get_gpu_optimized_config(self, batch_size: int, n_envs: int, device: str) -> Dict[str, Any]:
        """
        Get GPU-optimized training configuration based on device and memory.
        
        Args:
            batch_size: Original batch size
            n_envs: Number of environments
            device: Device string
            
        Returns:
            Optimized configuration dictionary
        """
        if 'cuda' not in device:
            # CPU optimization - smaller batches
            return {
                'batch_size': max(32, batch_size // 2),
                'n_envs': min(n_envs, 4)
            }
        
        # GPU optimization based on memory
        gpu_memory_gb = self.device_info.get('gpu_memory_gb', 4.0)
        
        if gpu_memory_gb <= 4.0:  # Your GTX 1050 Ti case
            # Conservative settings for 4GB GPU
            optimized_batch_size = max(64, min(batch_size, 128))
            optimized_n_envs = min(n_envs, 8)
        elif gpu_memory_gb <= 8.0:
            # Mid-range GPU
            optimized_batch_size = min(batch_size * 2, 256)
            optimized_n_envs = min(n_envs * 2, 16)
        else:
            # High-end GPU
            optimized_batch_size = batch_size * 2
            optimized_n_envs = n_envs * 2
            
        return {
            'batch_size': optimized_batch_size,
            'n_envs': optimized_n_envs
        }
        
    def create_environment(self,
                          env_id: str = "UAVTrajectoryOpt-v0",
                          n_envs: int = 1,
                          env_kwargs: Dict = None) -> gym.Env:
        """
        Create training environment(s).
        
        Args:
            env_id: Environment ID for registration
            n_envs: Number of parallel environments
            env_kwargs: Environment configuration
            
        Returns:
            Environment or vectorized environment
        """
        env_kwargs = env_kwargs or {}
        
        # Extract action_space_type from env_kwargs to avoid duplication
        # Make a copy to avoid modifying the original dict
        env_kwargs_copy = env_kwargs.copy()
        action_space_type = env_kwargs_copy.pop("action_space_type", "continuous")
        
        def _make_env():
            env = create_uav_env(
                params=self.params,
                action_space_type=action_space_type,
                max_episode_steps=300,
                **env_kwargs_copy
            )
            env = Monitor(env)
            return env
            
        if n_envs == 1:
            return _make_env()
        else:
            # Create vectorized environment
            return DummyVecEnv([_make_env for _ in range(n_envs)])
            
    def train_ppo(self, 
                  total_timesteps: int = 100000,
                  learning_rate: float = 3e-4,
                  n_steps: int = 2048,
                  batch_size: int = 64,
                  n_epochs: int = 10,
                  gamma: float = 0.99,
                  gae_lambda: float = 0.95,
                  clip_range: float = 0.2,
                  ent_coef: float = 0.0,
                  vf_coef: float = 0.5,
                  max_grad_norm: float = 0.5,
                  n_envs: int = 4) -> Tuple[PPO, Dict]:
        """
        Train PPO (Proximal Policy Optimization) algorithm.
        
        Args:
            total_timesteps: Total training timesteps
            learning_rate: Learning rate
            n_steps: Steps per environment per update
            batch_size: Minibatch size
            n_epochs: Number of epochs per update
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            clip_range: Clipping parameter
            ent_coef: Entropy coefficient
            vf_coef: Value function coefficient
            max_grad_norm: Max gradient norm
            n_envs: Number of parallel environments
            
        Returns:
            Trained model and metrics
        """
        print("Training PPO algorithm...")
        
        # Create environments
        train_env = self.create_environment(n_envs=n_envs)
        eval_env = self.create_environment(n_envs=1)

        # Create PPO model with conditional TensorBoard logging
        model_kwargs = {
            "policy": "MlpPolicy",
            "env": train_env,
            "learning_rate": learning_rate,
            "n_steps": n_steps,
            "batch_size": batch_size,
            "n_epochs": n_epochs,
            "gamma": gamma,
            "gae_lambda": gae_lambda,
            "clip_range": clip_range,
            "ent_coef": ent_coef,
            "vf_coef": vf_coef,
            "max_grad_norm": max_grad_norm,
            "verbose": 1,
            "device": self.device  # Use GPU as requested
        }
        
        # Add TensorBoard logging only if available
        if self.tensorboard_available:
            model_kwargs["tensorboard_log"] = f"{self.results_dir}/tensorboard/"
        
        model = PPO(**model_kwargs)
        
        # Setup callbacks
        metrics_callback = TrainingMetricsCallback(
            eval_env=eval_env,
            eval_freq=max(1000, total_timesteps // 20),
            n_eval_episodes=5
        )
        
        checkpoint_callback = CheckpointCallback(
            save_freq=max(5000, total_timesteps // 10),
            save_path=f"{self.results_dir}/ppo_checkpoints/",
            name_prefix="ppo_model"
        )
        
        # Train model
        model.learn(
            total_timesteps=total_timesteps,
            callback=[metrics_callback, checkpoint_callback],
            progress_bar=True
        )
        
        # Save final model
        model_path = f"{self.results_dir}/ppo_final_model"
        model.save(model_path)
        
        # Store results
        self.models['PPO'] = model
        self.training_metrics['PPO'] = metrics_callback.get_metrics()
        
        # Save metrics
        with open(f"{self.results_dir}/ppo_metrics.pkl", 'wb') as f:
            pickle.dump(self.training_metrics['PPO'], f)
            
        print(f"PPO training completed. Model saved to {model_path}")
        
        return model, self.training_metrics['PPO']
        
    def train_sac(self,
                  total_timesteps: int = 100000,
                  learning_rate: float = 3e-4,
                  buffer_size: int = 1000000,
                  learning_starts: int = 100,
                  batch_size: int = 256,
                  tau: float = 0.005,
                  gamma: float = 0.99,
                  train_freq: int = 1,
                  gradient_steps: int = 1,
                  ent_coef: str = "auto",
                  target_update_interval: int = 1) -> Tuple[SAC, Dict]:
        """
        Train SAC (Soft Actor-Critic) algorithm.
        
        Args:
            total_timesteps: Total training timesteps
            learning_rate: Learning rate
            buffer_size: Replay buffer size
            learning_starts: Timesteps before training starts
            batch_size: Minibatch size
            tau: Target network update rate
            gamma: Discount factor
            train_freq: Training frequency
            gradient_steps: Gradient steps per update
            ent_coef: Entropy coefficient
            target_update_interval: Target network update interval
            
        Returns:
            Trained model and metrics
        """
        print("Training SAC algorithm...")
        
        # Create environments
        train_env = self.create_environment(n_envs=1)
        eval_env = self.create_environment(n_envs=1)
        
        # Create SAC model with conditional TensorBoard logging
        model_kwargs = {
            "policy": "MlpPolicy",
            "env": train_env,
            "learning_rate": learning_rate,
            "buffer_size": buffer_size,
            "learning_starts": learning_starts,
            "batch_size": batch_size,
            "tau": tau,
            "gamma": gamma,
            "train_freq": train_freq,
            "gradient_steps": gradient_steps,
            "ent_coef": ent_coef,
            "target_update_interval": target_update_interval,
            "verbose": 1,
            "device": self.device
        }
        
        # Add TensorBoard logging only if available
        if self.tensorboard_available:
            model_kwargs["tensorboard_log"] = f"{self.results_dir}/tensorboard/"
        
        model = SAC(**model_kwargs)
        
        # Setup callbacks
        metrics_callback = TrainingMetricsCallback(
            eval_env=eval_env,
            eval_freq=max(1000, total_timesteps // 20),
            n_eval_episodes=5
        )
        
        checkpoint_callback = CheckpointCallback(
            save_freq=max(5000, total_timesteps // 10),
            save_path=f"{self.results_dir}/sac_checkpoints/",
            name_prefix="sac_model"
        )
        
        # Train model
        model.learn(
            total_timesteps=total_timesteps,
            callback=[metrics_callback, checkpoint_callback],
            progress_bar=True
        )
        
        # Save final model
        model_path = f"{self.results_dir}/sac_final_model"
        model.save(model_path)
        
        # Store results
        self.models['SAC'] = model
        self.training_metrics['SAC'] = metrics_callback.get_metrics()
        
        # Save metrics
        with open(f"{self.results_dir}/sac_metrics.pkl", 'wb') as f:
            pickle.dump(self.training_metrics['SAC'], f)
            
        print(f"SAC training completed. Model saved to {model_path}")
        
        return model, self.training_metrics['SAC']
        
    def train_dqn(self,
                  total_timesteps: int = 100000,
                  learning_rate: float = 1e-4,
                  buffer_size: int = 1000000,
                  learning_starts: int = 50000,
                  batch_size: int = 32,
                  tau: float = 1.0,
                  gamma: float = 0.99,
                  train_freq: int = 4,
                  gradient_steps: int = 1,
                  target_update_interval: int = 10000,
                  exploration_fraction: float = 0.1,
                  exploration_initial_eps: float = 1.0,
                  exploration_final_eps: float = 0.05) -> Tuple[DQN, Dict]:
        """
        Train DQN (Deep Q-Network) algorithm.
        Note: DQN requires discrete action space.
        
        Args:
            total_timesteps: Total training timesteps
            learning_rate: Learning rate
            buffer_size: Replay buffer size
            learning_starts: Timesteps before training starts
            batch_size: Minibatch size
            tau: Target network update rate
            gamma: Discount factor
            train_freq: Training frequency
            gradient_steps: Gradient steps per update
            target_update_interval: Target network update interval
            exploration_fraction: Fraction of timesteps for exploration
            exploration_initial_eps: Initial exploration rate
            exploration_final_eps: Final exploration rate
            
        Returns:
            Trained model and metrics
        """
        print("Training DQN algorithm...")
        
        # Create environments with discrete action space
        train_env = self.create_environment(
            n_envs=1, 
            env_kwargs={"action_space_type": "discrete"}
        )
        eval_env = self.create_environment(
            n_envs=1,
            env_kwargs={"action_space_type": "discrete"}
        )
        
        # Create DQN model with conditional TensorBoard logging
        model_kwargs = {
            "policy": "MlpPolicy",
            "env": train_env,
            "learning_rate": learning_rate,
            "buffer_size": buffer_size,
            "learning_starts": learning_starts,
            "batch_size": batch_size,
            "tau": tau,
            "gamma": gamma,
            "train_freq": train_freq,
            "gradient_steps": gradient_steps,
            "target_update_interval": target_update_interval,
            "exploration_fraction": exploration_fraction,
            "exploration_initial_eps": exploration_initial_eps,
            "exploration_final_eps": exploration_final_eps,
            "verbose": 1,
            "device": self.device
        }
        
        # Add TensorBoard logging only if available
        if self.tensorboard_available:
            model_kwargs["tensorboard_log"] = f"{self.results_dir}/tensorboard/"
        
        model = DQN(**model_kwargs)
        
        # Setup callbacks
        metrics_callback = TrainingMetricsCallback(
            eval_env=eval_env,
            eval_freq=max(1000, total_timesteps // 20),
            n_eval_episodes=5
        )
        
        checkpoint_callback = CheckpointCallback(
            save_freq=max(5000, total_timesteps // 10),
            save_path=f"{self.results_dir}/dqn_checkpoints/",
            name_prefix="dqn_model"
        )
        
        # Train model
        model.learn(
            total_timesteps=total_timesteps,
            callback=[metrics_callback, checkpoint_callback],
            progress_bar=True
        )
        
        # Save final model
        model_path = f"{self.results_dir}/dqn_final_model"
        model.save(model_path)
        
        # Store results
        self.models['DQN'] = model
        self.training_metrics['DQN'] = metrics_callback.get_metrics()
        
        # Save metrics
        with open(f"{self.results_dir}/dqn_metrics.pkl", 'wb') as f:
            pickle.dump(self.training_metrics['DQN'], f)
            
        print(f"DQN training completed. Model saved to {model_path}")
        
        return model, self.training_metrics['DQN']
        
    def evaluate_model(self, 
                      model_name: str,
                      n_eval_episodes: int = 100,
                      render: bool = False) -> Dict[str, Any]:
        """
        Evaluate a trained model.
        
        Args:
            model_name: Name of model to evaluate ('PPO', 'SAC', 'DQN')
            n_eval_episodes: Number of episodes for evaluation
            render: Whether to render episodes
            
        Returns:
            Evaluation results
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found. Available models: {list(self.models.keys())}")
            
        model = self.models[model_name]
        
        # Create evaluation environment
        action_space_type = "discrete" if model_name == "DQN" else "continuous"
        eval_env = self.create_environment(
            n_envs=1,
            env_kwargs={"action_space_type": action_space_type, "render_mode": "human" if render else None}
        )
        
        # Evaluate policy
        episode_rewards, episode_lengths = evaluate_policy(
            model, eval_env, n_eval_episodes=n_eval_episodes, return_episode_rewards=True
        )
        
        # Collect additional metrics
        episode_throughputs = []
        episode_distances_to_target = []
        
        for _ in range(min(10, n_eval_episodes)):  # Sample a few episodes for detailed metrics
            obs, _ = eval_env.reset()
            episode_throughput = 0
            done = False
            
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = eval_env.step(action)
                done = terminated or truncated
                
                if 'sum_throughput' in info:
                    episode_throughput += info['sum_throughput']
                    
                if done:
                    episode_throughputs.append(episode_throughput)
                    episode_distances_to_target.append(info['distance_to_target'])
                    
                if render:
                    eval_env.render()
        
        results = {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'mean_length': np.mean(episode_lengths),
            'std_length': np.std(episode_lengths),
            'mean_throughput': np.mean(episode_throughputs) if episode_throughputs else 0,
            'std_throughput': np.std(episode_throughputs) if episode_throughputs else 0,
            'mean_distance_to_target': np.mean(episode_distances_to_target) if episode_distances_to_target else 0,
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths,
            'episode_throughputs': episode_throughputs
        }
        
        return results
        
    def plot_training_curves(self, algorithms: List[str] = None, save_path: str = None):
        """
        Plot training convergence curves.
        
        Args:
            algorithms: List of algorithms to plot
            save_path: Path to save the plot
        """
        if algorithms is None:
            algorithms = list(self.training_metrics.keys())
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('RL Training Convergence Analysis', fontsize=16)
        
        colors = {'PPO': 'blue', 'SAC': 'red', 'DQN': 'green'}
        
        for algo in algorithms:
            if algo not in self.training_metrics:
                continue
                
            metrics = self.training_metrics[algo]
            color = colors.get(algo, 'black')
            
            # Plot evaluation rewards
            if metrics['evaluations_timesteps'] and metrics['evaluations_results']:
                axes[0, 0].plot(metrics['evaluations_timesteps'], metrics['evaluations_results'], 
                              label=algo, color=color, linewidth=2)
                
            # Plot episode rewards (smoothed)
            if metrics['episode_rewards']:
                # Smooth the rewards
                window_size = max(1, len(metrics['episode_rewards']) // 50)
                if len(metrics['episode_rewards']) >= window_size:
                    smoothed_rewards = np.convolve(metrics['episode_rewards'], 
                                                 np.ones(window_size)/window_size, mode='valid')
                    x_vals = np.arange(len(smoothed_rewards))
                    axes[0, 1].plot(x_vals, smoothed_rewards, label=algo, color=color, linewidth=2)
                    
            # Plot episode lengths
            if metrics['episode_lengths']:
                window_size = max(1, len(metrics['episode_lengths']) // 50)
                if len(metrics['episode_lengths']) >= window_size:
                    smoothed_lengths = np.convolve(metrics['episode_lengths'],
                                                 np.ones(window_size)/window_size, mode='valid')
                    x_vals = np.arange(len(smoothed_lengths))
                    axes[1, 0].plot(x_vals, smoothed_lengths, label=algo, color=color, linewidth=2)
                    
            # Plot episode throughputs
            if metrics['episode_throughputs']:
                window_size = max(1, len(metrics['episode_throughputs']) // 50)
                if len(metrics['episode_throughputs']) >= window_size:
                    smoothed_throughputs = np.convolve(metrics['episode_throughputs'],
                                                     np.ones(window_size)/window_size, mode='valid')
                    x_vals = np.arange(len(smoothed_throughputs))
                    axes[1, 1].plot(x_vals, smoothed_throughputs, label=algo, color=color, linewidth=2)
        
        # Customize plots
        axes[0, 0].set_title('Evaluation Rewards vs Timesteps')
        axes[0, 0].set_xlabel('Timesteps')
        axes[0, 0].set_ylabel('Mean Reward')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].set_title('Training Episode Rewards (Smoothed)')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Reward')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[1, 0].set_title('Episode Lengths (Smoothed)')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Length')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].set_title('Episode Throughputs (Smoothed)')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Sum Throughput')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training curves saved to {save_path}")
            
        plt.show()

def main():
    """Main training execution"""
    print("ELEC9123 Design Task F - Phase 2B: RL Algorithms Integration")
    print("=" * 60)
    
    # Create trainer
    trainer = UAVRLTrainer(results_dir="rl_training_results")
    
    # Train different algorithms
    print("Starting RL algorithm training...")
    
    # Train PPO (fast training for demonstration)
    ppo_model, ppo_metrics = trainer.train_ppo(total_timesteps=20000, n_envs=2)
    
    # Train SAC
    sac_model, sac_metrics = trainer.train_sac(total_timesteps=20000)
    
    # Train DQN
    dqn_model, dqn_metrics = trainer.train_dqn(total_timesteps=20000)
    
    # Plot training curves
    trainer.plot_training_curves(save_path="rl_training_results/training_curves.png")
    
    # Evaluate models
    print("\nEvaluating trained models...")
    for algo in ['PPO', 'SAC', 'DQN']:
        results = trainer.evaluate_model(algo, n_eval_episodes=20)
        print(f"\n{algo} Evaluation Results:")
        print(f"  Mean Reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
        print(f"  Mean Length: {results['mean_length']:.1f} ± {results['std_length']:.1f}")
        print(f"  Mean Throughput: {results['mean_throughput']:.2f} ± {results['std_throughput']:.2f}")
        print(f"  Mean Distance to Target: {results['mean_distance_to_target']:.1f}")
    
    print("\nRL training and evaluation completed successfully!")

if __name__ == "__main__":
    main()
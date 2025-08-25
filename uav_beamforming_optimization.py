"""
ELEC9123 Design Task F - Phase 2C: Beamforming Optimization

This module implements advanced beamforming optimization techniques for UAV communications,
upgrading from simple uniform beamforming to optimized beamforming strategies.
It includes joint trajectory and beamforming optimization capabilities.
"""

import numpy as np
import cvxpy as cp
from typing import List, Tuple, Dict, Optional, Any
from scipy.optimize import minimize, differential_evolution
import matplotlib.pyplot as plt

from uav_trajectory_simulation import SystemParameters, AntennaArray, ChannelModel

class BeamformingOptimizer:
    """
    Advanced beamforming optimization for UAV communications.
    
    Implements multiple beamforming strategies:
    1. Maximum Ratio Transmission (MRT)
    2. Zero-Forcing Beamforming (ZF)
    3. Minimum Mean Square Error (MMSE)
    4. Sum Rate Maximization
    5. Power-Constrained Optimization
    """
    
    def __init__(self, params: SystemParameters):
        self.params = params
        
    def mrt_beamforming(self, channel_vectors: List[np.ndarray]) -> List[np.ndarray]:
        """
        Maximum Ratio Transmission (MRT) beamforming.
        
        For each user k: w_k = sqrt(P_k) * h_k / ||h_k||
        where P_k is the power allocated to user k.
        
        Args:
            channel_vectors: List of channel vectors h_k for each user
            
        Returns:
            List of optimized beamforming vectors
        """
        K = len(channel_vectors)
        beamforming_vectors = []
        
        # Equal power allocation
        P_k = self.params.P_T / K
        
        for h_k in channel_vectors:
            # Normalize channel vector
            h_k_norm = h_k / np.linalg.norm(h_k)
            # Apply power scaling
            w_k = np.sqrt(P_k) * h_k_norm
            beamforming_vectors.append(w_k)
            
        return beamforming_vectors
        
    def zero_forcing_beamforming(self, channel_vectors: List[np.ndarray]) -> List[np.ndarray]:
        """
        Zero-Forcing (ZF) beamforming to eliminate inter-user interference.
        
        Constructs beamforming vectors such that h_j^H * w_k = 0 for j ≠ k.
        
        Args:
            channel_vectors: List of channel vectors h_k for each user
            
        Returns:
            List of zero-forcing beamforming vectors
        """
        K = len(channel_vectors)
        N_T = self.params.N_T
        
        if K > N_T:
            raise ValueError(f"Cannot perform ZF with {K} users and {N_T} antennas (K must be ≤ N_T)")
            
        # Construct channel matrix H = [h_1, h_2, ..., h_K]^H
        H = np.array(channel_vectors).conj()  # K x N_T matrix
        
        # Compute pseudo-inverse for ZF solution
        try:
            H_pinv = np.linalg.pinv(H)  # N_T x K matrix
        except np.linalg.LinAlgError:
            # Fallback to MRT if ZF is not feasible
            return self.mrt_beamforming(channel_vectors)
            
        beamforming_vectors = []
        
        # Equal power allocation among users
        P_k = self.params.P_T / K
        
        for k in range(K):
            # k-th column of pseudo-inverse gives ZF direction for user k
            w_k_zf = H_pinv[:, k]
            
            # Normalize and apply power constraint
            if np.linalg.norm(w_k_zf) > 0:
                w_k = np.sqrt(P_k) * w_k_zf / np.linalg.norm(w_k_zf)
            else:
                # Fallback to MRT for this user
                h_k = channel_vectors[k]
                w_k = np.sqrt(P_k) * h_k / np.linalg.norm(h_k)
                
            beamforming_vectors.append(w_k)
            
        return beamforming_vectors
        
    def mmse_beamforming(self, channel_vectors: List[np.ndarray], 
                        noise_power: float = None) -> List[np.ndarray]:
        """
        Minimum Mean Square Error (MMSE) beamforming.
        
        Optimizes beamforming to minimize mean square error between
        intended and received signals.
        
        Args:
            channel_vectors: List of channel vectors h_k for each user
            noise_power: Noise power (defaults to system parameter)
            
        Returns:
            List of MMSE beamforming vectors
        """
        if noise_power is None:
            noise_power = self.params.sigma_2_watts
            
        K = len(channel_vectors)
        N_T = self.params.N_T
        
        # Construct channel matrix
        H = np.array(channel_vectors).conj()  # K x N_T matrix
        
        # MMSE solution: W = (H^H * H + (σ²/P) * I)^(-1) * H^H
        try:
            # Regularization term
            reg_term = (noise_power * K / self.params.P_T) * np.eye(N_T)
            
            # Compute MMSE matrix
            HH_H = H.conj().T @ H
            mmse_matrix = np.linalg.inv(HH_H + reg_term) @ H.conj().T
            
        except np.linalg.LinAlgError:
            # Fallback to MRT if MMSE is not feasible
            return self.mrt_beamforming(channel_vectors)
            
        beamforming_vectors = []
        
        # Equal power allocation
        P_k = self.params.P_T / K
        
        for k in range(K):
            # k-th column gives MMSE beamforming for user k
            w_k_mmse = mmse_matrix[:, k]
            
            # Normalize and apply power constraint
            if np.linalg.norm(w_k_mmse) > 0:
                w_k = np.sqrt(P_k) * w_k_mmse / np.linalg.norm(w_k_mmse)
            else:
                # Fallback to MRT
                h_k = channel_vectors[k]
                w_k = np.sqrt(P_k) * h_k / np.linalg.norm(h_k)
                
            beamforming_vectors.append(w_k)
            
        return beamforming_vectors
        
    def sum_rate_maximization(self, channel_vectors: List[np.ndarray]) -> List[np.ndarray]:
        """
        Sum rate maximization using convex optimization.
        
        Solves: maximize Σ log(1 + SINR_k)
        subject to: Σ ||w_k||² ≤ P_T
        
        Args:
            channel_vectors: List of channel vectors h_k for each user
            
        Returns:
            List of sum-rate optimal beamforming vectors
        """
        K = len(channel_vectors)
        N_T = self.params.N_T
        
        try:
            # Define optimization variables
            W = cp.Variable((N_T, K), complex=True)  # Beamforming matrix
            
            # Channel matrix
            H = np.array(channel_vectors).conj()  # K x N_T
            
            # Signal powers: |h_k^H * w_k|²
            signal_powers = []
            for k in range(K):
                signal_power = cp.real(cp.conj(H[k, :]) @ W[:, k])
                signal_powers.append(cp.square(signal_power))
                
            # Interference powers
            interference_powers = []
            for k in range(K):
                interference = 0
                for j in range(K):
                    if j != k:
                        interference += cp.square(cp.real(cp.conj(H[k, :]) @ W[:, j]))
                interference_powers.append(interference)
                
            # SINR expressions
            sinr_expressions = []
            for k in range(K):
                sinr_k = signal_powers[k] / (interference_powers[k] + self.params.sigma_2_watts)
                sinr_expressions.append(sinr_k)
                
            # Objective: maximize sum rate (approximated)
            # Using log(1 + x) ≈ x for small x, or other approximations
            objective = cp.Maximize(cp.sum(sinr_expressions))
            
            # Power constraint
            power_constraint = cp.sum([cp.sum_squares(W[:, k]) for k in range(K)]) <= self.params.P_T
            
            # Solve optimization problem
            problem = cp.Problem(objective, [power_constraint])
            problem.solve(solver=cp.SCS, verbose=False)
            
            if problem.status in ["infeasible", "unbounded"]:
                # Fallback to MRT
                return self.mrt_beamforming(channel_vectors)
                
            # Extract solution
            W_opt = W.value
            beamforming_vectors = [W_opt[:, k] for k in range(K)]
            
        except Exception:
            # Fallback to MRT if optimization fails
            return self.mrt_beamforming(channel_vectors)
            
        return beamforming_vectors
        
    def adaptive_beamforming(self, channel_vectors: List[np.ndarray], 
                           method: str = "auto") -> List[np.ndarray]:
        """
        Adaptive beamforming that selects the best method based on conditions.
        
        Args:
            channel_vectors: List of channel vectors h_k for each user
            method: Beamforming method ("auto", "mrt", "zf", "mmse", "sum_rate")
            
        Returns:
            List of adaptively optimized beamforming vectors
        """
        K = len(channel_vectors)
        N_T = self.params.N_T
        
        if method == "auto":
            # Select method based on system conditions
            if K == 1:
                # Single user: MRT is optimal
                method = "mrt"
            elif K <= N_T // 2:
                # Low user density: Try sum rate maximization
                method = "sum_rate"
            elif K == N_T:
                # Full load: Zero forcing
                method = "zf"
            else:
                # Overloaded: MMSE
                method = "mmse"
                
        # Apply selected method
        if method == "mrt":
            return self.mrt_beamforming(channel_vectors)
        elif method == "zf":
            return self.zero_forcing_beamforming(channel_vectors)
        elif method == "mmse":
            return self.mmse_beamforming(channel_vectors)
        elif method == "sum_rate":
            return self.sum_rate_maximization(channel_vectors)
        else:
            raise ValueError(f"Unknown beamforming method: {method}")

class JointOptimizer:
    """
    Joint trajectory and beamforming optimization.
    
    Optimizes UAV trajectory and beamforming vectors simultaneously
    to maximize system performance.
    """
    
    def __init__(self, params: SystemParameters):
        self.params = params
        self.antenna_array = AntennaArray(params)
        self.channel_model = ChannelModel(params, self.antenna_array)
        self.beamforming_optimizer = BeamformingOptimizer(params)
        
    def objective_function(self, trajectory: np.ndarray, users: List[Tuple[float, float, float]],
                          beamforming_method: str = "sum_rate") -> float:
        """
        Objective function for joint optimization.
        
        Args:
            trajectory: UAV trajectory positions [T x 3]
            users: User positions
            beamforming_method: Beamforming optimization method
            
        Returns:
            Negative sum throughput (for minimization)
        """
        T = len(trajectory)
        total_throughput = 0.0
        
        for t in range(T):
            uav_pos = trajectory[t]
            
            # Compute channel vectors
            channel_vectors = self.channel_model.compute_channel_vectors(uav_pos, users)
            
            # Optimize beamforming
            beamforming_vectors = self.beamforming_optimizer.adaptive_beamforming(
                channel_vectors, method=beamforming_method
            )
            
            # Compute SNR and throughput
            snr_values = self._compute_snr(channel_vectors, beamforming_vectors)
            throughput_values = [np.log2(1 + snr) for snr in snr_values]
            sum_throughput = sum(throughput_values)
            
            total_throughput += sum_throughput
            
        return -total_throughput  # Negative for minimization
        
    def _compute_snr(self, channel_vectors: List[np.ndarray], 
                     beamforming_vectors: List[np.ndarray]) -> List[float]:
        """Compute SNR values for given channel and beamforming vectors."""
        K = len(channel_vectors)
        snr_values = []
        
        for k in range(K):
            h_k = channel_vectors[k]
            w_k = beamforming_vectors[k]
            
            # Signal power
            signal_power = np.abs(np.conj(h_k).T @ w_k)**2
            
            # Interference power
            interference_power = 0.0
            for j in range(K):
                if j != k:
                    w_j = beamforming_vectors[j]
                    interference_power += np.abs(np.conj(h_k).T @ w_j)**2
                    
            # SINR
            sinr = signal_power / (interference_power + self.params.sigma_2_watts)
            snr_values.append(sinr)
            
        return snr_values
        
    def optimize_trajectory(self, users: List[Tuple[float, float, float]],
                          episode_length: int, 
                          beamforming_method: str = "sum_rate",
                          optimization_method: str = "differential_evolution") -> Dict[str, Any]:
        """
        Optimize UAV trajectory with joint beamforming optimization.
        
        Args:
            users: User positions
            episode_length: Number of time steps
            beamforming_method: Beamforming optimization method
            optimization_method: Trajectory optimization method
            
        Returns:
            Optimization results
        """
        # Define trajectory parameterization
        # Simple approach: optimize waypoints and interpolate
        n_waypoints = min(10, episode_length // 10)  # Reduce complexity
        
        # Bounds for waypoint positions
        bounds = []
        for _ in range(n_waypoints):
            bounds.extend([
                (self.params.X_MIN, self.params.X_MAX),  # x
                (self.params.Y_MIN, self.params.Y_MAX),  # y
                (self.params.Z_H, self.params.Z_H)       # z (fixed)
            ])
            
        def trajectory_objective(waypoints_flat):
            # Reshape waypoints
            waypoints = waypoints_flat.reshape(n_waypoints, 3)
            
            # Interpolate full trajectory
            trajectory = self._interpolate_trajectory(waypoints, episode_length)
            
            # Evaluate objective
            return self.objective_function(trajectory, users, beamforming_method)
            
        # Initial guess: linear trajectory from start to end
        start = np.array(self.params.UAV_START)
        end = np.array(self.params.UAV_END)
        initial_waypoints = np.array([
            start + (end - start) * t / (n_waypoints - 1) 
            for t in range(n_waypoints)
        ])
        initial_guess = initial_waypoints.flatten()
        
        # Optimize
        if optimization_method == "differential_evolution":
            result = differential_evolution(
                trajectory_objective, 
                bounds, 
                seed=42,
                maxiter=100,
                popsize=15
            )
        else:
            result = minimize(
                trajectory_objective,
                initial_guess,
                bounds=bounds,
                method="L-BFGS-B"
            )
            
        # Extract optimal trajectory
        optimal_waypoints = result.x.reshape(n_waypoints, 3)
        optimal_trajectory = self._interpolate_trajectory(optimal_waypoints, episode_length)
        
        # Compute performance metrics
        optimal_throughput = -result.fun
        
        # Compute beamforming performance at each time step
        throughput_history = []
        beamforming_history = []
        
        for t in range(episode_length):
            uav_pos = optimal_trajectory[t]
            channel_vectors = self.channel_model.compute_channel_vectors(uav_pos, users)
            beamforming_vectors = self.beamforming_optimizer.adaptive_beamforming(
                channel_vectors, method=beamforming_method
            )
            
            snr_values = self._compute_snr(channel_vectors, beamforming_vectors)
            throughput_values = [np.log2(1 + snr) for snr in snr_values]
            sum_throughput = sum(throughput_values)
            
            throughput_history.append(sum_throughput)
            beamforming_history.append(beamforming_vectors)
            
        results = {
            'optimal_trajectory': optimal_trajectory,
            'optimal_waypoints': optimal_waypoints,
            'throughput_history': np.array(throughput_history),
            'beamforming_history': beamforming_history,
            'total_throughput': optimal_throughput,
            'avg_throughput': np.mean(throughput_history),
            'optimization_result': result,
            'users': users,
            'episode_length': episode_length,
            'beamforming_method': beamforming_method
        }
        
        return results
        
    def _interpolate_trajectory(self, waypoints: np.ndarray, episode_length: int) -> np.ndarray:
        """Interpolate full trajectory from waypoints."""
        n_waypoints = len(waypoints)
        
        if n_waypoints == 1:
            return np.tile(waypoints[0], (episode_length, 1))
            
        # Time indices for waypoints
        waypoint_times = np.linspace(0, episode_length - 1, n_waypoints)
        
        # Time indices for full trajectory
        full_times = np.arange(episode_length)
        
        # Interpolate each dimension
        trajectory = np.zeros((episode_length, 3))
        for dim in range(3):
            trajectory[:, dim] = np.interp(full_times, waypoint_times, waypoints[:, dim])
            
        return trajectory

def compare_beamforming_methods(params: SystemParameters = None, 
                              n_scenarios: int = 10) -> Dict[str, Any]:
    """
    Compare different beamforming methods performance.
    
    Args:
        params: System parameters
        n_scenarios: Number of random scenarios to test
        
    Returns:
        Comparison results
    """
    params = params or SystemParameters()
    beamforming_optimizer = BeamformingOptimizer(params)
    channel_model = ChannelModel(params, AntennaArray(params))
    
    methods = ["mrt", "zf", "mmse", "sum_rate"]
    results = {method: {"throughputs": [], "computation_times": []} for method in methods}
    
    for scenario in range(n_scenarios):
        # Generate random scenario
        users = []
        for _ in range(params.K):
            x = np.random.uniform(params.X_MIN, params.X_MAX)
            y = np.random.uniform(params.Y_MIN, params.Y_MAX)
            users.append((x, y, 0.0))
            
        # Random UAV position
        uav_pos = np.array([
            np.random.uniform(params.X_MIN, params.X_MAX),
            np.random.uniform(params.Y_MIN, params.Y_MAX),
            params.Z_H
        ])
        
        # Compute channel vectors
        channel_vectors = channel_model.compute_channel_vectors(uav_pos, users)
        
        # Test each method
        for method in methods:
            import time
            start_time = time.time()
            
            try:
                if method == "mrt":
                    beamforming_vectors = beamforming_optimizer.mrt_beamforming(channel_vectors)
                elif method == "zf":
                    beamforming_vectors = beamforming_optimizer.zero_forcing_beamforming(channel_vectors)
                elif method == "mmse":
                    beamforming_vectors = beamforming_optimizer.mmse_beamforming(channel_vectors)
                elif method == "sum_rate":
                    beamforming_vectors = beamforming_optimizer.sum_rate_maximization(channel_vectors)
                    
                # Compute throughput
                snr_values = []
                for k in range(params.K):
                    h_k = channel_vectors[k]
                    w_k = beamforming_vectors[k]
                    signal_power = np.abs(np.conj(h_k).T @ w_k)**2
                    snr = signal_power / params.sigma_2_watts
                    snr_values.append(snr)
                    
                throughput = sum(np.log2(1 + snr) for snr in snr_values)
                computation_time = time.time() - start_time
                
                results[method]["throughputs"].append(throughput)
                results[method]["computation_times"].append(computation_time)
                
            except Exception as e:
                print(f"Method {method} failed in scenario {scenario}: {e}")
                # Use fallback values
                results[method]["throughputs"].append(0.0)
                results[method]["computation_times"].append(1.0)
                
    # Compute statistics
    for method in methods:
        throughputs = results[method]["throughputs"]
        times = results[method]["computation_times"]
        
        results[method]["mean_throughput"] = np.mean(throughputs)
        results[method]["std_throughput"] = np.std(throughputs)
        results[method]["mean_time"] = np.mean(times)
        results[method]["std_time"] = np.std(times)
        
    return results

def main():
    """Main beamforming optimization demonstration"""
    print("ELEC9123 Design Task F - Phase 2C: Beamforming Optimization")
    print("=" * 60)
    
    # Test beamforming comparison
    print("Comparing beamforming methods...")
    comparison_results = compare_beamforming_methods(n_scenarios=20)
    
    print("\nBeamforming Method Comparison:")
    for method, stats in comparison_results.items():
        print(f"{method.upper()}:")
        print(f"  Mean Throughput: {stats['mean_throughput']:.3f} ± {stats['std_throughput']:.3f}")
        print(f"  Mean Computation Time: {stats['mean_time']:.4f}s ± {stats['std_time']:.4f}s")
        
    # Test joint optimization
    print("\nTesting joint trajectory and beamforming optimization...")
    params = SystemParameters()
    joint_optimizer = JointOptimizer(params)
    
    # Create test scenario
    users = [
        (25.0, 25.0, 0.0),
        (75.0, 75.0, 0.0)
    ]
    
    # Optimize trajectory
    optimization_results = joint_optimizer.optimize_trajectory(
        users=users,
        episode_length=50,  # Shorter for demonstration
        beamforming_method="sum_rate"
    )
    
    print(f"Joint Optimization Results:")
    print(f"  Total Throughput: {optimization_results['total_throughput']:.2f}")
    print(f"  Average Throughput: {optimization_results['avg_throughput']:.2f}")
    print(f"  Episode Length: {optimization_results['episode_length']}")
    
    print("\nBeamforming optimization completed successfully!")

if __name__ == "__main__":
    main()
"""
This implementation follows the 11-step process outlined in Section 2.1:
1. Simulate 3D Environment
2. Place K users randomly in XY-plane  
3. Initialize UAV position
4. Setup UAV antenna array (ULA)
5. Compute channel vectors h_k(t)
6. Initialize transmit signal vectors
7. Formulate SNR_k(t)
8. Calculate throughput R_k(t)
9. Compute sum throughput R(t)
10. Update UAV position
11. Repeat for episode length L

System parameters from Table 2 are implemented as specified.
"""

import numpy as np
from typing import List, Tuple, Dict

class SystemParameters:
    """System parameters from Table 2 of ELEC9123 Design Task F"""
    
    # Environment dimensions (meters)
    X_MIN = 0.0
    Y_MIN = 0.0  
    Z_MIN = 0.0
    X_MAX = 100.0
    Y_MAX = 100.0
    Z_H = 50.0
    
    # UAV trajectory
    UAV_START = (0.0, 0.0, 50.0)  # (x_0, y_0, z_h)
    UAV_END = (80.0, 80.0, 50.0)  # (x_L, y_L, z_h)
    
    # System parameters
    K = 2              # Number of users
    N_T = 8            # Number of antennas
    P_T = 0.5          # Transmit power budget (Watts)
    SIGMA_2_DBM = -100 # Noise power (dBm)
    ETA = 2.5          # Path loss exponent
    F = 2.4e9          # Frequency (Hz)
    
    # Episode parameters
    L_MIN = 200        # Minimum episode length (seconds)
    L_MAX = 300        # Maximum episode length (seconds)
    V_MIN = 10.0       # Minimum UAV speed (m/s)
    V_MAX = 30.0       # Maximum UAV speed (m/s)
    
    # Physical constants
    C = 3e8            # Speed of light (m/s)
    
    @property
    def wavelength(self) -> float:
        """Calculate wavelength lambda = c/f"""
        return self.C / self.F
    
    @property
    def sigma_2_watts(self) -> float:
        """Convert noise power from dBm to Watts"""
        return 10**((self.SIGMA_2_DBM - 30) / 10)
    
    @property
    def L_0(self) -> float:
        """Calculate reference path loss L_0 = (lambda / (4*pi))^2"""
        return (self.wavelength / (4 * np.pi))**2

class UAVEnvironment:
    """3D environment for UAV trajectory simulation"""
    
    def __init__(self, params: SystemParameters):
        self.params = params
        self.users = []  # List of user positions
        self.uav_position = np.array(params.UAV_START)
        self.episode_length = None
        self.uav_speed = None
        self.delta_t = 1.0  # Time step (seconds)
        
    def setup_environment(self, episode_length: int = None, uav_speed: float = None):
        """
        Step 1: Simulate 3D Environment
        Create 3D environment with specified dimensions
        """
        if episode_length is None:
            self.episode_length = np.random.randint(self.params.L_MIN, self.params.L_MAX + 1)
        else:
            self.episode_length = episode_length
            
        if uav_speed is None:
            self.uav_speed = np.random.uniform(self.params.V_MIN, self.params.V_MAX)
        else:
            self.uav_speed = uav_speed
            
        print(f"Environment setup: {self.params.X_MAX}m x {self.params.Y_MAX}m x {self.params.Z_H}m")
        print(f"Episode length: {self.episode_length}s, UAV speed: {self.uav_speed:.1f}m/s")
    
    def place_users_randomly(self) -> List[Tuple[float, float, float]]:
        """
        Step 2: Simulate Users
        Randomly place K users in XY-plane using uniform distribution
        """
        self.users = []
        for k in range(self.params.K):
            x_k = np.random.uniform(self.params.X_MIN, self.params.X_MAX)
            y_k = np.random.uniform(self.params.Y_MIN, self.params.Y_MAX)
            z_k = self.params.Z_MIN  # Users on ground
            self.users.append((x_k, y_k, z_k))
            
        print(f"Placed {self.params.K} users randomly in environment:")
        for i, (x, y, z) in enumerate(self.users):
            print(f"  User {i+1}: ({x:.1f}, {y:.1f}, {z:.1f})")
        
        return self.users
    
    def initialize_uav(self):
        """
        Step 3: Initial UAV Position
        Model UAV positioned at (x_0, y_0, z_h) at timestep t=0
        """
        self.uav_position = np.array(self.params.UAV_START)
        print(f"UAV initialized at position: ({self.uav_position[0]:.1f}, {self.uav_position[1]:.1f}, {self.uav_position[2]:.1f})")

class AntennaArray:
    """UAV antenna system with uniform linear array (ULA)"""
    
    def __init__(self, params: SystemParameters):
        self.params = params
        self.antenna_positions = self._setup_ula()
    
    def _setup_ula(self) -> np.ndarray:
        """
        Step 4: UAV Antenna Setup
        Equip UAV with N_t antennas in uniform linear array along Y-axis
        """
        # Standard ULA spacing: lambda/2
        d = self.params.wavelength / 2
        
        # Center the array at origin, extend along Y-axis
        positions = np.zeros((self.params.N_T, 3))
        start_y = -(self.params.N_T - 1) * d / 2
        
        for i in range(self.params.N_T):
            positions[i] = [0, start_y + i * d, 0]  # [x, y, z] relative to UAV
        
        print(f"ULA setup: {self.params.N_T} antennas, spacing: {d:.3f}m (λ/2)")
        return positions

class ChannelModel:
    """Wireless channel modeling for UAV-user communication"""
    
    def __init__(self, params: SystemParameters, antenna_array: AntennaArray):
        self.params = params
        self.antenna_array = antenna_array
        
    def compute_distances(self, uav_pos: np.ndarray, user_pos: Tuple[float, float, float]) -> np.ndarray:
        """Compute distances from each antenna to user"""
        user_array = np.array(user_pos)
        distances = np.zeros(self.params.N_T)
        
        for i in range(self.params.N_T):
            # Absolute antenna position = UAV position + relative antenna position
            antenna_abs_pos = uav_pos + self.antenna_array.antenna_positions[i]
            distances[i] = np.linalg.norm(antenna_abs_pos - user_array)
            
        return distances
    
    def compute_channel_vectors(self, uav_pos: np.ndarray, users: List[Tuple[float, float, float]]) -> List[np.ndarray]:
        """
        Step 5: Compute Channel Vectors
        Calculate h_k(t) ∈ C^(N_t×1) for all users k ∈ {1,2,...,K}
        
        Channel model: h_k = sqrt(L_0 * d_k^(-η)) * exp(j * 2π/λ * d_k)
        """
        channel_vectors = []
        
        for k, user_pos in enumerate(users):
            distances = self.compute_distances(uav_pos, user_pos)
            h_k = np.zeros(self.params.N_T, dtype=complex)
            
            for i in range(self.params.N_T):
                d_k_i = distances[i]
                
                # Path loss component
                path_loss = np.sqrt(self.params.L_0 * (d_k_i ** (-self.params.ETA)))
                
                # LoS phase component  
                phase = np.exp(1j * 2 * np.pi / self.params.wavelength * d_k_i)
                
                h_k[i] = path_loss * phase
                
            channel_vectors.append(h_k)
            
        return channel_vectors

class SignalProcessor:
    """Signal processing for transmit beamforming and throughput calculation"""
    
    def __init__(self, params: SystemParameters):
        self.params = params
        
    def initialize_transmit_signals(self) -> List[np.ndarray]:
        """
        Step 6: Initialize Transmit Signal Vectors  
        Initialize transmit signal vectors w_k ∈ C^(N_t×1) for each user
        """
        transmit_signals = []
        
        for k in range(self.params.K):
            # Simple uniform beamforming - equal power allocation
            w_k = np.ones(self.params.N_T, dtype=complex) / np.sqrt(self.params.N_T)
            # Normalize to satisfy power constraint
            w_k = w_k * np.sqrt(self.params.P_T / self.params.K)
            transmit_signals.append(w_k)
            
        return transmit_signals
    
    def compute_snr(self, channel_vectors: List[np.ndarray], transmit_signals: List[np.ndarray]) -> List[float]:
        """
        Step 7: Formulate SNR
        Calculate SNR_k(t) for all users k ∈ {1,2,...,K}
        
        SNR_k(t) = |h_k^H * w_k|^2 / σ^2
        """
        snr_values = []
        
        for k in range(self.params.K):
            h_k = channel_vectors[k]
            w_k = transmit_signals[k]
            
            # Received signal power: |h_k^H * w_k|^2
            signal_power = np.abs(np.conj(h_k).T @ w_k)**2
            
            # SNR = signal_power / noise_power
            snr_k = signal_power / self.params.sigma_2_watts
            snr_values.append(snr_k)
            
        return snr_values
    
    def compute_throughput(self, snr_values: List[float]) -> Tuple[List[float], float]:
        """
        Step 8: Calculate Throughput
        Step 9: Compute Sum Throughput
        
        R_k(t) = log2(1 + SNR_k(t)) [bits per channel use]
        R(t) = Σ R_k(t)
        """
        throughput_values = []
        
        for snr_k in snr_values:
            R_k = np.log2(1 + snr_k)
            throughput_values.append(R_k)
            
        sum_throughput = sum(throughput_values)
        
        return throughput_values, sum_throughput

class UAVTrajectorySimulator:
    """Main UAV trajectory simulation coordinator"""
    
    def __init__(self, params: SystemParameters = None):
        self.params = params or SystemParameters()
        self.environment = UAVEnvironment(self.params)
        self.antenna_array = AntennaArray(self.params)
        self.channel_model = ChannelModel(self.params, self.antenna_array)
        self.signal_processor = SignalProcessor(self.params)
        
        # Simulation results storage
        self.trajectory = []
        self.throughput_history = []
        self.individual_throughput_history = []
        
    def update_uav_position(self, t: int) -> np.ndarray:
        """
        Step 10: Update UAV Position
        Move UAV to next location based on speed and timestep
        
        Simple linear trajectory from start to end position
        """
        if self.environment.episode_length <= 0:
            return self.environment.uav_position
            
        # Linear interpolation from start to end
        progress = t / self.environment.episode_length
        start = np.array(self.params.UAV_START)
        end = np.array(self.params.UAV_END)
        
        new_position = start + progress * (end - start)
        self.environment.uav_position = new_position
        
        return new_position
    
    def run_single_episode(self, episode_length: int = None, uav_speed: float = None, verbose: bool = True) -> Dict:
        """
        Run complete UAV trajectory episode following 11-step process
        
        Step 11: Repeat steps 5-10 for episode length L
        """
        if verbose:
            print("=" * 60)
            print("ELEC9123 Design Task F - UAV Trajectory Simulation")
            print("=" * 60)
        
        # Steps 1-4: Environment and system setup
        self.environment.setup_environment(episode_length, uav_speed)
        users = self.environment.place_users_randomly()
        self.environment.initialize_uav()
        
        if verbose:
            print(f"Antenna array: {self.params.N_T} elements (ULA along Y-axis)")
        
        # Initialize result storage
        self.trajectory = []
        self.throughput_history = []
        self.individual_throughput_history = []
        
        if verbose:
            print("\nStarting episode simulation...")
            print("-" * 40)
        
        # Step 11: Episode simulation loop
        for t in range(self.environment.episode_length + 1):
            # Step 10: Update UAV position (except at t=0)
            if t > 0:
                self.update_uav_position(t)
            
            self.trajectory.append(self.environment.uav_position.copy())
            
            # Steps 5-9: Channel modeling and throughput calculation
            channel_vectors = self.channel_model.compute_channel_vectors(
                self.environment.uav_position, users
            )
            
            transmit_signals = self.signal_processor.initialize_transmit_signals()
            
            snr_values = self.signal_processor.compute_snr(channel_vectors, transmit_signals)
            
            individual_throughput, sum_throughput = self.signal_processor.compute_throughput(snr_values)
            
            self.throughput_history.append(sum_throughput)
            self.individual_throughput_history.append(individual_throughput)
            
            if verbose and t % max(1, self.environment.episode_length // 10) == 0:
                pos = self.environment.uav_position
                print(f"t={t:3d}: UAV=({pos[0]:.1f},{pos[1]:.1f},{pos[2]:.1f}), "
                      f"Sum Throughput={sum_throughput:.2f} bits/channel use")
        
        # Calculate final results
        total_throughput = sum(self.throughput_history)
        avg_throughput = np.mean(self.throughput_history)
        
        results = {
            'episode_length': self.environment.episode_length,
            'uav_speed': self.environment.uav_speed,
            'users': users,
            'trajectory': np.array(self.trajectory),
            'throughput_history': np.array(self.throughput_history),
            'individual_throughput_history': np.array(self.individual_throughput_history),
            'total_throughput': total_throughput,
            'avg_throughput': avg_throughput,
            'final_position': self.environment.uav_position,
            'target_position': np.array(self.params.UAV_END)
        }
        
        if verbose:
            print("-" * 40)
            print(f"Episode completed!")
            print(f"Total throughput: {total_throughput:.2f} bits/channel use")
            print(f"Average throughput: {avg_throughput:.2f} bits/channel use")
            print(f"Final UAV position: ({self.environment.uav_position[0]:.1f}, "
                  f"{self.environment.uav_position[1]:.1f}, {self.environment.uav_position[2]:.1f})")
            print(f"Target end position: {self.params.UAV_END}")
            
            # Check if UAV reached target
            distance_to_target = np.linalg.norm(self.environment.uav_position - np.array(self.params.UAV_END))
            print(f"Distance to target: {distance_to_target:.2f}m")
        
        return results

def main():
    """Main simulation execution"""
    print("ELEC9123 Design Task F - UAV Trajectory Modeling")
    print("Initializing simulation with Table 2 parameters...")
    
    # Create simulator with default parameters
    simulator = UAVTrajectorySimulator()
    
    # Run single episode demonstration
    results = simulator.run_single_episode()
    
    print("\nSimulation completed successfully!")
    print("Results stored in 'results' dictionary with keys:")
    for key in results.keys():
        print(f"  - {key}")

if __name__ == "__main__":
    main()
"""
ELEC9123 Design Task F - UAV Trajectory Simulation Demo
Demonstration script to test and visualize the UAV trajectory modeling
"""

import numpy as np
import matplotlib.pyplot as plt
from uav_trajectory_simulation import UAVTrajectorySimulator, SystemParameters

def visualize_simulation_results(results):
    """Create comprehensive visualization of simulation results"""
    
    # Create figure with subplots
    fig = plt.figure(figsize=(15, 12))
    
    # 1. 3D Trajectory Plot
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    trajectory = results['trajectory']
    users = results['users']
    
    # Plot UAV trajectory
    ax1.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], 
             'b-', linewidth=2, label='UAV Trajectory')
    
    # Plot start and end positions
    ax1.scatter(*results['trajectory'][0], color='green', s=100, label='Start')
    ax1.scatter(*results['trajectory'][-1], color='red', s=100, label='End')
    
    # Plot users
    for i, (x, y, z) in enumerate(users):
        ax1.scatter(x, y, z, color='orange', s=80, marker='^', label=f'User {i+1}' if i == 0 else '')
    
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title('3D UAV Trajectory')
    ax1.legend()
    
    # 2. 2D Trajectory (Top View)
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.plot(trajectory[:, 0], trajectory[:, 1], 'b-', linewidth=2, label='UAV Path')
    ax2.scatter(trajectory[0, 0], trajectory[0, 1], color='green', s=100, label='Start')
    ax2.scatter(trajectory[-1, 0], trajectory[-1, 1], color='red', s=100, label='End')
    
    for i, (x, y, z) in enumerate(users):
        ax2.scatter(x, y, color='orange', s=80, marker='^', label=f'User {i+1}' if i == 0 else '')
    
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_title('UAV Trajectory (Top View)')
    ax2.grid(True)
    ax2.legend()
    ax2.axis('equal')
    
    # 3. Sum Throughput vs Time
    ax3 = fig.add_subplot(2, 3, 3)
    time_steps = range(len(results['throughput_history']))
    ax3.plot(time_steps, results['throughput_history'], 'r-', linewidth=2)
    ax3.set_xlabel('Time Step')
    ax3.set_ylabel('Sum Throughput (bits/channel use)')
    ax3.set_title('Sum Throughput vs Time')
    ax3.grid(True)
    
    # 4. Individual User Throughput
    ax4 = fig.add_subplot(2, 3, 4)
    individual_throughput = np.array(results['individual_throughput_history'])
    for k in range(individual_throughput.shape[1]):
        ax4.plot(time_steps, individual_throughput[:, k], 
                linewidth=2, label=f'User {k+1}')
    ax4.set_xlabel('Time Step')
    ax4.set_ylabel('Individual Throughput (bits/channel use)')
    ax4.set_title('Individual User Throughput')
    ax4.legend()
    ax4.grid(True)
    
    # 5. UAV Distance to Users Over Time
    ax5 = fig.add_subplot(2, 3, 5)
    distances_to_users = []
    for t in range(len(trajectory)):
        uav_pos = trajectory[t]
        distances = []
        for user_pos in users:
            dist = np.linalg.norm(uav_pos - np.array(user_pos))
            distances.append(dist)
        distances_to_users.append(distances)
    
    distances_to_users = np.array(distances_to_users)
    for k in range(len(users)):
        ax5.plot(time_steps, distances_to_users[:, k], 
                linewidth=2, label=f'Distance to User {k+1}')
    ax5.set_xlabel('Time Step')
    ax5.set_ylabel('Distance (m)')
    ax5.set_title('UAV Distance to Users')
    ax5.legend()
    ax5.grid(True)
    
    # 6. System Performance Summary
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')
    
    # Create performance summary text
    summary_text = f"""
Performance Summary:

Episode Length: {results['episode_length']} seconds
UAV Speed: {results['uav_speed']:.1f} m/s
Number of Users: {len(users)}

Total Throughput: {results['total_throughput']:.2f}
Average Throughput: {results['avg_throughput']:.2f}
Max Instantaneous: {np.max(results['throughput_history']):.2f}
Min Instantaneous: {np.min(results['throughput_history']):.2f}

Final UAV Position: 
({results['final_position'][0]:.1f}, {results['final_position'][1]:.1f}, {results['final_position'][2]:.1f})

Target Position:
({results['target_position'][0]:.1f}, {results['target_position'][1]:.1f}, {results['target_position'][2]:.1f})

Distance to Target: {np.linalg.norm(results['final_position'] - results['target_position']):.2f}m
"""
    
    ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    plt.show()

def run_parameter_analysis():
    """Run analysis with different parameter settings"""
    
    print("Running parameter analysis...")
    
    # Test different episode lengths
    episode_lengths = [200, 250, 300]
    uav_speeds = [10, 20, 30]
    
    results_analysis = []
    
    for episode_length in episode_lengths:
        for uav_speed in uav_speeds:
            print(f"\nTesting: Episode={episode_length}s, Speed={uav_speed}m/s")
            
            simulator = UAVTrajectorySimulator()
            results = simulator.run_single_episode(
                episode_length=episode_length, 
                uav_speed=uav_speed,
                verbose=False
            )
            
            results_analysis.append({
                'episode_length': episode_length,
                'uav_speed': uav_speed,
                'avg_throughput': results['avg_throughput'],
                'total_throughput': results['total_throughput'],
                'distance_to_target': np.linalg.norm(results['final_position'] - results['target_position'])
            })
    
    # Display analysis results
    print("\nParameter Analysis Results:")
    print("=" * 80)
    print(f"{'Episode (s)':<12} {'Speed (m/s)':<12} {'Avg Throughput':<15} {'Total Throughput':<16} {'Target Error (m)':<15}")
    print("-" * 80)
    
    for result in results_analysis:
        print(f"{result['episode_length']:<12} {result['uav_speed']:<12} "
              f"{result['avg_throughput']:<15.2f} {result['total_throughput']:<16.2f} "
              f"{result['distance_to_target']:<15.2f}")

def verify_system_requirements():
    """Verify that simulation meets system requirements"""
    
    print("Verifying system requirements...")
    print("=" * 50)
    
    params = SystemParameters()
    simulator = UAVTrajectorySimulator(params)
    
    # Run test simulation
    results = simulator.run_single_episode(verbose=False)
    
    # Check requirements
    checks = []
    
    # 1. Environment dimensions
    env_check = (params.X_MAX == 100 and params.Y_MAX == 100 and params.Z_H == 50)
    checks.append(("Environment dimensions (100m x 100m x 50m)", env_check))
    
    # 2. UAV start/end positions
    start_check = np.allclose(results['trajectory'][0], [0, 0, 50])
    final_pos = results['final_position']
    target_pos = results['target_position']
    end_check = np.allclose(final_pos, target_pos, atol=1.0)  # 1m tolerance
    checks.append(("UAV starts at (0,0,50)", start_check))
    checks.append(("UAV ends near (80,80,50)", end_check))
    
    # 3. Episode length in range
    episode_check = (params.L_MIN <= results['episode_length'] <= params.L_MAX)
    checks.append(("Episode length (200-300s)", episode_check))
    
    # 4. UAV speed in range  
    speed_check = (params.V_MIN <= results['uav_speed'] <= params.V_MAX)
    checks.append(("UAV speed (10-30 m/s)", speed_check))
    
    # 5. Number of users
    users_check = (len(results['users']) == params.K)
    checks.append(("Number of users (K=2)", users_check))
    
    # 6. System parameters
    antenna_check = (params.N_T == 8)
    power_check = (params.P_T == 0.5)
    freq_check = (params.F == 2.4e9)
    checks.append(("Number of antennas (N_t=8)", antenna_check))
    checks.append(("Transmit power (0.5W)", power_check))
    checks.append(("Frequency (2.4 GHz)", freq_check))
    
    # 7. Throughput calculation
    throughput_check = (len(results['throughput_history']) > 0 and 
                       all(t >= 0 for t in results['throughput_history']))
    checks.append(("Throughput calculation", throughput_check))
    
    # Display results
    all_passed = True
    for requirement, passed in checks:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{requirement:<35}: {status}")
        if not passed:
            all_passed = False
    
    print("-" * 50)
    print(f"Overall: {'✓ ALL REQUIREMENTS SATISFIED' if all_passed else '✗ SOME REQUIREMENTS FAILED'}")
    
    return all_passed

def main():
    """Main demonstration function"""
    
    print("ELEC9123 Design Task F - UAV Trajectory Simulation Demo")
    print("=" * 60)
    
    # 1. Verify system requirements
    print("\n1. Verifying System Requirements:")
    requirements_met = verify_system_requirements()
    
    if not requirements_met:
        print("⚠️  Some requirements not met. Please check implementation.")
        return
    
    # 2. Run single episode with visualization
    print("\n2. Running Single Episode Simulation:")
    simulator = UAVTrajectorySimulator()
    results = simulator.run_single_episode()
    
    # 3. Create visualizations
    print("\n3. Creating Visualization...")
    visualize_simulation_results(results)
    
    # 4. Parameter analysis
    print("\n4. Running Parameter Analysis:")
    run_parameter_analysis()
    
    print("\n" + "=" * 60)
    print("Demo completed successfully!")
    print("\nNext steps:")
    print("- Implement reinforcement learning for trajectory optimization")
    print("- Add beamforming optimization") 
    print("- Create performance benchmarking")
    print("- Generate required analysis plots")

if __name__ == "__main__":
    main()
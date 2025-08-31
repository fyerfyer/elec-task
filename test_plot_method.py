#!/usr/bin/env python3
"""
Test script to verify the plot_algorithm_analysis method
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from uav_rl_training import UAVRLTrainer
from uav_trajectory_simulation import SystemParameters

def test_plot_algorithm_analysis():
    """Test the plot_algorithm_analysis method"""
    print("🧪 Testing plot_algorithm_analysis method...")
    
    try:
        # Create trainer instance
        trainer = UAVRLTrainer(
            params=SystemParameters(),
            results_dir="test_results",
            verbose=False
        )
        
        # Create dummy training metrics for testing
        trainer.training_metrics['TEST'] = {
            'evaluations_timesteps': [1000, 2000, 3000, 4000, 5000],
            'evaluations_results': [-100, -80, -60, -40, -20],
            'episode_rewards': [-150, -120, -100, -80, -60, -50, -40, -30, -25, -20],
            'episode_lengths': [100, 120, 110, 105, 95, 90, 85, 88, 92, 95],
            'episode_throughputs': [1000, 1200, 1100, 1300, 1400, 1500, 1600, 1550, 1650, 1700]
        }
        
        # Test with valid algorithm
        print("Testing with valid algorithm data...")
        trainer.plot_algorithm_analysis('TEST', save_path="test_results/test_analysis.png")
        print("✅ Test with valid data passed")
        
        # Test with missing algorithm
        print("Testing with missing algorithm...")
        trainer.plot_algorithm_analysis('MISSING')
        print("✅ Test with missing algorithm passed")
        
        # Test with empty metrics
        print("Testing with empty metrics...")
        trainer.training_metrics['EMPTY'] = {
            'evaluations_timesteps': [],
            'evaluations_results': [],
            'episode_rewards': [],
            'episode_lengths': [],
            'episode_throughputs': []
        }
        trainer.plot_algorithm_analysis('EMPTY', save_path="test_results/empty_analysis.png")
        print("✅ Test with empty metrics passed")
        
        print("🎉 All tests passed! plot_algorithm_analysis method works correctly.")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_plot_algorithm_analysis()
    sys.exit(0 if success else 1)

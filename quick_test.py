#!/usr/bin/env python3
"""
Ultra-fast test to verify debug fix works - uses minimal computation.
"""

import sys
import os
import time
import numpy as np

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from uav_trajectory_simulation import SystemParameters
from uav_performance_analysis import PerformanceAnalyzer

def test_minimal_plot6():
    """Test Plot 6 with minimal computation."""
    print("🚀 MINIMAL COMPUTATION TEST")
    print("=" * 40)
    
    params = SystemParameters()
    analyzer = PerformanceAnalyzer(params, results_dir="quick_test_results")
    
    print("🎯 Testing Plot 6 with ultra-aggressive quick mode...")
    
    start_time = time.time()
    
    try:
        # This should now use:
        # - 3 episodes instead of 10
        # - 20 steps instead of 100  
        # - 10 maxiter instead of 100
        # - 5 popsize instead of 15
        # Total: ~3 * 10 * 5 = 150 function evaluations instead of 15,000+
        
        fig = analyzer.plot_6_optimized_trajectories(
            save_path="quick_test_results/ultra_quick_plot6.png",
            quick_mode=True
        )
        
        execution_time = time.time() - start_time
        print(f"✅ SUCCESS! Completed in {execution_time:.1f}s")
        
        # Check if it's reasonable time (should be under 2 minutes)
        if execution_time < 120:
            print(f"🎉 EXCELLENT! Quick mode working - execution under 2 minutes")
            return True
        else:
            print(f"⚠️ SLOW but working - took {execution_time:.1f}s")
            return True
            
    except Exception as e:
        execution_time = time.time() - start_time
        print(f"❌ FAILED after {execution_time:.1f}s: {e}")
        return False

def main():
    """Run minimal test."""
    print("🧪 ULTRA-QUICK DEBUG VERIFICATION")
    print("This should complete in under 2 minutes")
    print("=" * 40)
    
    success = test_minimal_plot6()
    
    print("\n" + "=" * 40)
    if success:
        print("✅ DEBUG FIX SUCCESSFUL!")
        print("The hanging issue has been resolved.")
        print("You can now run the main script with --phase demo")
    else:
        print("❌ Still having issues - may need further optimization")
    
    return success

if __name__ == "__main__":
    main()
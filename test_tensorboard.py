#!/usr/bin/env python3
"""
Quick test script to check TensorBoard installation
"""

def test_tensorboard_import():
    """Test if TensorBoard can be imported and initialized"""
    try:
        import tensorboard
        print(f"✅ TensorBoard version: {tensorboard.__version__}")
        
        # Test basic functionality
        from torch.utils.tensorboard import SummaryWriter
        import tempfile
        import os
        
        # Create a temporary directory for testing
        with tempfile.TemporaryDirectory() as temp_dir:
            writer = SummaryWriter(temp_dir)
            writer.add_scalar('test/dummy', 1.0, 0)
            writer.close()
            print("✅ TensorBoard SummaryWriter works correctly")
            
        return True
        
    except ImportError as e:
        print(f"❌ TensorBoard import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ TensorBoard functionality test failed: {e}")
        return False

if __name__ == "__main__":
    print("🔍 Testing TensorBoard installation...")
    success = test_tensorboard_import()
    
    if success:
        print("✅ TensorBoard is properly installed and functional")
    else:
        print("❌ TensorBoard installation issues detected")
        print("💡 Try: pip install tensorboard")

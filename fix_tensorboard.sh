#!/bin/bash

# Quick fix script for ELEC Task F TensorBoard issues
echo "🔧 Quick fix for TensorBoard installation..."

# Check if we're in the correct environment
if [[ "$CONDA_DEFAULT_ENV" != "elec_taskf" ]]; then
    echo "⚠️  Warning: Not in 'elec_taskf' conda environment"
    echo "Please run: conda activate elec_taskf"
    echo "Or continue with current environment? (y/N)"
    read -r response
    if [[ "$response" != "y" ]]; then
        exit 1
    fi
fi

# Install/upgrade TensorBoard and related packages
echo "📦 Installing TensorBoard and dependencies..."
pip install --upgrade tensorboard>=2.12.0
pip install --upgrade protobuf>=3.20.0
pip install --upgrade stable-baselines3>=2.0.0

# Test TensorBoard installation
echo "🔍 Testing TensorBoard installation..."
python -c "
import tensorboard
print(f'✅ TensorBoard version: {tensorboard.__version__}')

try:
    from torch.utils.tensorboard import SummaryWriter
    import tempfile
    import os
    
    with tempfile.TemporaryDirectory() as temp_dir:
        writer = SummaryWriter(temp_dir)
        writer.add_scalar('test/dummy', 1.0, 0)
        writer.close()
    print('✅ TensorBoard functionality test passed')
except Exception as e:
    print(f'❌ TensorBoard test failed: {e}')
    exit(1)
"

if [ $? -eq 0 ]; then
    echo "✅ TensorBoard is now properly installed!"
    echo "🚀 You can now run: python run_full_rl_training.py"
else
    echo "❌ TensorBoard installation still has issues"
    echo "💡 Try manual installation: pip install tensorboard"
fi

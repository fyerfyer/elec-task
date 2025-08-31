#!/bin/bash
"""
ELEC9123 Design Task F - Quick Setup Script
Installs missing dependencies for RL training
"""

echo "🔧 ELEC9123 TaskF - Dependency Check and Installation"
echo "=================================================="

# Check if we're in a conda environment
if [[ $CONDA_DEFAULT_ENV ]]; then
    echo "✅ Conda environment detected: $CONDA_DEFAULT_ENV"
else
    echo "⚠️  No conda environment active, using system Python"
fi

# Install TensorBoard if not present
echo "📦 Checking TensorBoard installation..."
if python -c "import tensorboard" 2>/dev/null; then
    echo "✅ TensorBoard already installed"
else
    echo "📥 Installing TensorBoard..."
    pip install tensorboard>=2.8.0
fi

# Install other potentially missing packages
echo "📦 Installing/upgrading key packages..."
pip install -r requirements.txt

echo ""
echo "✅ Setup complete! You can now run the training scripts."
echo "🚀 Run: python run_full_rl_training.py"

#!/bin/bash

# ELEC9123 Task F - Install and Update Dependencies
echo "🔧 Installing/Updating dependencies for ELEC9123 Task F..."

# Ensure we're in the conda environment
if [[ "$CONDA_DEFAULT_ENV" != "elec_taskf" ]]; then
    echo "⚠️  Warning: Not in 'elec_taskf' conda environment"
    echo "Please run: conda activate elec_taskf"
    exit 1
fi

echo "✅ Using conda environment: $CONDA_DEFAULT_ENV"

# Update pip first
echo "📦 Updating pip..."
pip install --upgrade pip

# Install/update core dependencies
echo "📦 Installing core dependencies..."
pip install torch>=2.0.0 torchvision>=0.15.0 torchaudio>=2.0.0 --index-url https://download.pytorch.org/whl/cu121

# Install RL dependencies
echo "📦 Installing RL dependencies..."
pip install stable-baselines3>=2.0.0
pip install gymnasium>=0.28.0

# Install TensorBoard specifically
echo "📦 Installing TensorBoard..."
pip install tensorboard>=2.12.0

# Install other scientific computing packages
echo "📦 Installing scientific computing packages..."
pip install numpy>=1.24.0 matplotlib>=3.7.0 scipy>=1.11.0
pip install pandas>=2.0.0 seaborn>=0.12.0 scikit-learn>=1.3.0
pip install plotly>=5.14.0 cvxpy>=1.2.0

# Verify installations
echo "🔍 Verifying installations..."
python -c "import torch; print(f'PyTorch: {torch.__version__} (CUDA: {torch.cuda.is_available()})')"
python -c "import stable_baselines3; print(f'Stable-Baselines3: {stable_baselines3.__version__}')"
python -c "import tensorboard; print(f'TensorBoard: {tensorboard.__version__}')"
python -c "import gymnasium; print(f'Gymnasium: {gymnasium.__version__}')"

echo "✅ Dependencies installation completed!"
echo "💡 You can now run: python run_full_rl_training.py"

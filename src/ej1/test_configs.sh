#!/bin/bash
# test_configs.sh
# Tests multiple autoencoder configurations and saves comparative outputs.

# Go to the project root directory
cd "$(dirname "$0")/../../"

VENV_DIR=".venv"

# Create venv if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
fi

# Activate venv
source "$VENV_DIR/bin/activate"

# Check NumPy (install if missing)
if ! python3 -c "import numpy" &> /dev/null; then
    echo "Installing dependencies..."
    pip install numpy matplotlib --quiet
fi

export PYTHONPATH="$(pwd):$PYTHONPATH"

# Array of configurations to test
# Format: "name learning_rate optimizer hidden_activation encoder_layers decoder_layers"
configs=(
    "default 0.01 adam tanh 35,16,8,2 2,8,16,35"
    "momentum_opt 0.01 momentum tanh 35,16,8,2 2,8,16,35"
    "relu_activation 0.01 adam relu 35,16,8,2 2,8,16,35"
    "deep_network 0.01 adam tanh 35,16,10,5,2 2,5,10,16,35"
    "high_lr 0.1 adam tanh 35,16,8,2 2,8,16,35"
    "low_lr 0.001 adam tanh 35,16,8,2 2,8,16,35"
    "shallow_network 0.01 adam tanh 35,8,2 2,8,35"
    "asymmetric 0.01 adam tanh 35,16,8,2 2,16,35"
)

# Output directory for all tests
mkdir -p results_comparative

for config in "${configs[@]}"; do
    read -r name lr opt act enc dec <<< "$config"
    
    echo "=========================================================="
    echo "Testing configuration: $name"
    echo "LR: $lr | Optimizer: $opt | Activation: $act"
    echo "Encoder: $enc | Decoder: $dec"
    echo "=========================================================="
    
    # Train
    python3 src/ej1/train.py --lr "$lr" --optimizer "$opt" --hidden_act "$act" --encoder "$enc" --decoder "$dec"
    
    # Visualize (generates graphs and new characters)
    python3 src/ej1/visualize.py --lr "$lr" --optimizer "$opt" --hidden_act "$act" --encoder "$enc" --decoder "$dec"
    
    # Create directory for the current configuration
    mkdir -p "results_comparative/$name"
    
    # Move outputs into the config-specific directory
    if [ -d "outputs" ]; then
        mv outputs/* "results_comparative/$name/"
    fi
    
    echo "Results for $name saved in results_comparative/$name/"
    echo ""
done

echo "=========================================================="
echo "Generating final comparative graph of all latent spaces..."
python3 src/ej1/compare_latent.py

echo "=========================================================="
echo "All tests completed!"
echo "Check 'results_comparative/' for comparative graphs and generated letters."

#!/bin/bash
# Variational Autoencoder — test runner for ej2
# Usage: ./ej2/run_tests.sh [command]

# Anchor to ej2/ directory
cd "$(dirname "$0")" || exit 1

# Setup Virtual Environment
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment at .venv/..."
    python3 -m venv .venv
fi

source .venv/bin/activate

# Install dependencies if needed
if ! python3 -c "import numpy, matplotlib" &> /dev/null; then
    echo "Installing dependencies (numpy, matplotlib)..."
    pip install --quiet numpy matplotlib
fi

# Helper functions for output
print_ok() {
    echo -e "\033[32m✓ $1\033[0m"
}

print_err() {
    echo -e "\033[31m✗ $1\033[0m"
}

run_train() {
    echo "Training VAE..."
    if python3 src/train_vae.py; then
        print_ok "Training completed successfully."
    else
        print_err "Training failed."
        exit 1
    fi
}

run_visualize() {
    echo "Generating Visualizations..."
    if python3 src/visualize_vae.py; then
        print_ok "Visualizations generated in outputs/."
    else
        print_err "Visualization generation failed."
        exit 1
    fi
}

run_generate() {
    local n=${1:-16}
    echo "Generating $n samples..."
    if python3 src/visualize_vae.py --generate "$n"; then
        print_ok "Samples generated successfully."
    else
        print_err "Sample generation failed."
        exit 1
    fi
}

run_test() {
    echo "Evaluating reconstruction quality..."
    # The train_vae script prints the evaluation at the end, so we can just train for 0 epochs if we had a dedicated test script.
    # We will run python3 -c to do a quick evaluation.
    if python3 -c "
import sys, os
sys.path.append(os.path.abspath('src'))
from visualize_vae import load_model
from dataset import load_emoji_dataset
import numpy as np

outputs_dir = 'outputs'
x, labels = load_emoji_dataset()
vae = load_model(outputs_dir)
x_hat, _, _, _ = vae.forward(x)
x_hat_bin = (x_hat > 0.5).astype(int)
x_bin = x.astype(int)
errors = np.sum(x_hat_bin != x_bin)
total = x.shape[0] * x.shape[1]
print(f'Total Pixel Errors: {errors} / {total}')
"; then
        print_ok "Reconstruction evaluated."
    else
        print_err "Evaluation failed."
        exit 1
    fi
}

run_preview() {
    echo "Previewing Emoji Dataset..."
    python3 -c "import sys; sys.path.append('src'); from dataset import preview_dataset; preview_dataset()"
    print_ok "Preview completed."
}

run_beta_sweep() {
    echo "Running Beta Sweep (0.5, 1.0, 2.0, 4.0)..."
    rm -f outputs/beta_sweep.csv
    for beta in 0.5 1.0 2.0 4.0; do
        echo "Training with beta=$beta..."
        python3 src/train_vae.py --beta="$beta" --sweep beta_sweep.csv > /dev/null
        print_ok "Beta=$beta completed."
    done
    echo "Beta sweep results saved to outputs/beta_sweep.csv"
}

run_latent_sweep() {
    echo "Running Latent Dim Sweep (2, 4, 8)..."
    rm -f outputs/latent_sweep.csv
    # We'll temporarily modify config_vae.py or pass a flag to train_vae.py
    # But train_vae.py doesn't instantiate with CLI latent_dim directly. Let's use sed on config if we must.
    # Actually wait, I can just use sed to change LATENT_DIM in config.
    cp src/config_vae.py src/config_vae_backup.py
    for dim in 2 4 8; do
        echo "Training with latent_dim=$dim..."
        sed -i "s/^LATENT_DIM = .*/LATENT_DIM = $dim/g" src/config_vae.py
        python3 src/train_vae.py --sweep latent_sweep.csv > /dev/null
        print_ok "Latent_dim=$dim completed."
    done
    mv src/config_vae_backup.py src/config_vae.py
    echo "Latent sweep results saved to outputs/latent_sweep.csv"
}

run_clean() {
    echo "Cleaning outputs/ directory..."
    rm -f outputs/*.png outputs/*.csv outputs/*.npz
    print_ok "Outputs cleaned."
}

CMD=${1:-"all"}

case "$CMD" in
    train) run_train ;;
    visualize) run_visualize ;;
    generate) run_generate "$2" ;;
    test) run_test ;;
    preview) run_preview ;;
    beta-sweep) run_beta_sweep ;;
    latent-sweep) run_latent_sweep ;;
    clean) run_clean ;;
    all)
        run_clean
        run_train
        run_visualize
        run_test
        echo "======================================"
        print_ok "ALL COMMANDS EXECUTED SUCCESSFULLY!"
        echo "Results are available in ej2/outputs/"
        ;;
    *)
        echo "Unknown command: $CMD"
        echo "Usage: $0 [train|visualize|generate N|test|preview|beta-sweep|latent-sweep|clean|all]"
        exit 1
        ;;
esac

#!/bin/bash
# Usage: ./run_tests.sh [command]
cd "$(dirname "$0")"

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

command=$1

case $command in
    train)
        echo "Training autoencoder..."
        export PYTHONPATH=.
        python3 src/train.py
        ;;
    visualize)
        echo "Generating visualizations..."
        export PYTHONPATH=.
        python3 src/visualize.py
        ;;
    test)
        echo "Running pixel error test..."
        export PYTHONPATH=.
        python3 src/visualize.py --test
        ;;
    preview)
        echo "Previewing font characters..."
        export PYTHONPATH=.
        python3 src/font_loader.py
        ;;
    sweep)
        echo "Sweeping is a manual process right now, but you can edit src/config.py to test different lr and epochs."
        ;;
    clean)
        echo "Cleaning outputs directory..."
        rm -rf outputs/*
        touch outputs/.gitkeep
        echo "Cleaned."
        ;;
    all)
        echo "[1/3] Training autoencoder..."
        export PYTHONPATH=.
        python3 src/train.py
        echo ""
        echo "[2/3] Generating visualizations..."
        python3 src/visualize.py
        echo ""
        echo "[3/3] Pixel error test..."
        python3 src/visualize.py --test
        ;;
    *)
        echo "Usage: ./run_tests.sh [train|visualize|test|preview|sweep|clean|all]"
        ;;
esac
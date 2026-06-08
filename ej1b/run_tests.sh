#!/bin/bash
# Denoising Autoencoder — test runner
# Usage: ./ej1b/run_tests.sh [command]
#
# Commands:
#   train              - Train the DAE with default config
#   test               - Evaluate reconstruction at all noise levels
#   test-noise N       - Test at a specific noise level N (e.g. 0.20)
#   visualize          - Generate all plots
#   noise-sweep        - Train and evaluate across all noise levels automatically
#   compare-noise-fns  - Train 3 models (one per noise function) and compare
#   clean              - Remove ej1b/outputs/*
#   all                - train + visualize + test

# Change to the directory of this script
cd "$(dirname "$0")" || exit 1

VENV_DIR=".venv"

echo "=== DAE Test Runner ==="

# Set up virtual environment and install dependencies if not exists
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment in $VENV_DIR..."
    python3 -m venv "$VENV_DIR"
    source "$VENV_DIR/bin/activate"
    echo "Installing requirements (numpy, matplotlib)..."
    pip install --quiet numpy matplotlib
else
    source "$VENV_DIR/bin/activate"
fi

COMMAND=$1

case $COMMAND in
    train)
        echo "Training DAE..."
        python -m src.train_dae
        if [ $? -eq 0 ]; then echo "✓ Training completed"; else echo "✗ Training failed"; exit 1; fi
        ;;
    test)
        echo "Evaluating on all noise levels..."
        python -m src.train_dae --test-only
        if [ $? -eq 0 ]; then echo "✓ Testing completed"; else echo "✗ Testing failed"; exit 1; fi
        ;;
    test-noise)
        if [ -z "$2" ]; then
            echo "✗ Missing noise level argument N for test-noise"
            exit 1
        fi
        echo "Testing at noise level $2..."
        python -m src.visualize_dae --noise-level "$2"
        if [ $? -eq 0 ]; then echo "✓ Test for noise $2 completed"; else echo "✗ Test failed"; exit 1; fi
        ;;
    visualize)
        echo "Generating plots..."
        python -m src.visualize_dae
        if [ $? -eq 0 ]; then echo "✓ Visualization completed"; else echo "✗ Visualization failed"; exit 1; fi
        ;;
    noise-sweep)
        echo "Running noise sweep..."
        python -m src.train_dae --sweep
        if [ $? -eq 0 ]; then echo "✓ Noise sweep completed"; else echo "✗ Noise sweep failed"; exit 1; fi
        ;;
    compare-noise-fns)
        echo "Comparing noise functions..."
        python -m src.train_dae --compare-fns
        if [ $? -eq 0 ]; then echo "✓ Comparison completed"; else echo "✗ Comparison failed"; exit 1; fi
        ;;
    clean)
        echo "Cleaning outputs..."
        rm -rf outputs/*
        touch outputs/.gitkeep
        echo "✓ Cleaned outputs"
        ;;
    all)
        echo "Running full pipeline (train, visualize, test)..."
        ./run_tests.sh train
        if [ $? -ne 0 ]; then exit 1; fi
        ./run_tests.sh visualize
        if [ $? -ne 0 ]; then exit 1; fi
        ./run_tests.sh test
        if [ $? -ne 0 ]; then exit 1; fi
        echo "✓ All tasks completed successfully"
        ;;
    *)
        echo "Usage: $0 [train|test|test-noise N|visualize|noise-sweep|compare-noise-fns|clean|all]"
        exit 1
        ;;
esac

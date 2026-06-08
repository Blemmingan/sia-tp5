"""
train_dae.py

Script to train the Denoising Autoencoder (DAE) and evaluate its performance
across various noise levels.
"""
import sys
import os
import argparse
import numpy as np
import csv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from ej1.src.font_loader import load_font
from ej1.src.optimizers import get_optimizer
from src.dae import DAE
from src.noise import salt_and_pepper, add_gaussian_noise, random_pixel_dropout
import src.config_dae as config

def get_noise_fn(name):
    if name == "salt_and_pepper":
        return salt_and_pepper
    elif name == "gaussian":
        return add_gaussian_noise
    elif name == "dropout":
        return random_pixel_dropout
    else:
        raise ValueError(f"Unknown noise function: {name}")

def evaluate_noise_levels(dae, clean_data, noise_fn_name, noise_levels, output_csv=None):
    """
    Evaluates the DAE on a list of noise levels and returns a summary.
    """
    noise_fn = get_noise_fn(noise_fn_name)
    results = []
    
    print(f"\nEvaluating with {noise_fn_name} noise:")
    print("Noise Level | Avg Pixel Error | Max Pixel Error | % Chars Within Threshold")
    print("-" * 75)
    
    for level in noise_levels:
        # Generate noisy data
        noisy_data = noise_fn(clean_data, level)
        
        # Reconstruct
        reconstructed = dae.reconstruct(noisy_data)
        
        # Calculate errors per character
        # Convert predictions to binary (threshold 0.5) to match pixels
        binary_preds = (reconstructed >= 0.5).astype(int)
        errors_per_char = np.sum(np.abs(binary_preds - clean_data), axis=1)
        
        avg_error = np.mean(errors_per_char)
        max_error = np.max(errors_per_char)
        pct_within_threshold = np.mean(errors_per_char <= config.MAX_PIXEL_ERROR) * 100
        
        results.append({
            "Noise Level": level,
            "Avg Pixel Error": avg_error,
            "Max Pixel Error": max_error,
            "% Chars Within Threshold": pct_within_threshold
        })
        
        print(f"{level:<11.2f} | {avg_error:<15.2f} | {max_error:<15} | {pct_within_threshold:>23.0f}%")
        
    if output_csv:
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        with open(output_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"Results saved to {output_csv}")
        
    return results

def train_model(dae, clean_data, noise_fn_name, noise_level, epochs, opt_name, lr):
    """
    Trains the DAE by dynamically adding noise each epoch.
    """
    noise_fn = get_noise_fn(noise_fn_name)
    optimizer = get_optimizer(opt_name, lr)
    
    print(f"Training DAE for {epochs} epochs with {noise_fn_name} at {noise_level} noise level...")
    
    for epoch in range(1, epochs + 1):
        # Generate fresh noisy data each epoch
        noisy_data = noise_fn(clean_data, noise_level)
        
        # Forward pass with noisy data
        dae.forward(noisy_data)
        
        # Backward pass with clean data
        grads, loss = dae.backward(clean_data, loss_type=config.LOSS)
        
        # Update weights
        optimizer.update(dae.params, grads)
        
        if epoch % 500 == 0 or epoch == 1:
            print(f"Epoch {epoch:04d}/{epochs} - Loss: {loss:.4f}")
            
    return dae

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-only", action="store_true", help="Evaluate existing weights")
    parser.add_argument("--sweep", action="store_true", help="Sweep across noise levels")
    parser.add_argument("--compare-fns", action="store_true", help="Compare noise functions")
    args = parser.parse_args()
    
    clean_data = load_font(os.path.abspath(os.path.join(os.path.dirname(__file__), '../font.h')))
    
    os.makedirs(os.path.join(os.path.dirname(__file__), '../outputs'), exist_ok=True)
    weights_path = os.path.join(os.path.dirname(__file__), '../outputs/dae_weights.npz')
    csv_path = os.path.join(os.path.dirname(__file__), '../outputs/noise_study.csv')
    
    if args.compare_fns:
        functions = ["salt_and_pepper", "gaussian", "dropout"]
        for fn in functions:
            dae = DAE(config.ENCODER_LAYERS, config.DECODER_LAYERS)
            dae = train_model(dae, clean_data, fn, config.TRAIN_NOISE_LEVEL, config.EPOCHS, config.OPTIMIZER, config.LEARNING_RATE)
            evaluate_noise_levels(dae, clean_data, fn, config.NOISE_LEVELS, 
                                  output_csv=csv_path.replace(".csv", f"_{fn}.csv"))
        return
        
    if args.sweep:
        all_results = []
        for level in config.NOISE_LEVELS:
            dae = DAE(config.ENCODER_LAYERS, config.DECODER_LAYERS)
            dae = train_model(dae, clean_data, config.NOISE_FUNCTION, level, config.EPOCHS, config.OPTIMIZER, config.LEARNING_RATE)
            res = evaluate_noise_levels(dae, clean_data, config.NOISE_FUNCTION, [level])[0]
            all_results.append(res)
            
        with open(csv_path.replace(".csv", "_sweep.csv"), 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
            writer.writeheader()
            writer.writerows(all_results)
        return

    dae = DAE(config.ENCODER_LAYERS, config.DECODER_LAYERS)
    
    if args.test_only:
        if os.path.exists(weights_path):
            data = np.load(weights_path, allow_pickle=True)
            dae.params = list(data['params'])
        else:
            print("No weights found, using random initialized weights.")
            
        evaluate_noise_levels(dae, clean_data, config.NOISE_FUNCTION, config.NOISE_LEVELS, output_csv=csv_path)
    else:
        dae = train_model(dae, clean_data, config.NOISE_FUNCTION, config.TRAIN_NOISE_LEVEL, config.EPOCHS, config.OPTIMIZER, config.LEARNING_RATE)
        
        # Save weights
        # Save list of numpy arrays into npz
        np.savez(weights_path, params=np.array(dae.params, dtype=object))
        print(f"Weights saved to {weights_path}")
        
        evaluate_noise_levels(dae, clean_data, config.NOISE_FUNCTION, config.NOISE_LEVELS, output_csv=csv_path)

if __name__ == "__main__":
    main()

"""
visualize_dae.py

Generates visualizations for the DAE including reconstruction grids and error plots.
"""
import sys
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from ej1.src.font_loader import load_font
from src.dae import DAE
from src.train_dae import get_noise_fn
import src.config_dae as config

def plot_noise_levels_grid(dae, clean_data, noise_fn_name, noise_levels, output_path):
    """
    Grid with rows = noise levels, columns = characters (sample 8 chars).
    Each cell shows: noisy input | reconstructed | original.
    """
    noise_fn = get_noise_fn(noise_fn_name)
    sample_indices = np.linspace(0, 31, 8, dtype=int)
    num_samples = len(sample_indices)
    
    fig, axes = plt.subplots(len(noise_levels), num_samples, figsize=(2 * num_samples, 2.5 * len(noise_levels)))
    # If only 1 noise level, axes is 1D
    if len(noise_levels) == 1:
        axes = np.expand_dims(axes, 0)
        
    for i, level in enumerate(noise_levels):
        noisy_data = noise_fn(clean_data, level)
        reconstructed = dae.reconstruct(noisy_data)
        
        for j, idx in enumerate(sample_indices):
            ax = axes[i, j]
            
            # Combine 3 images horizontally: noisy | reconstructed | original
            orig_img = clean_data[idx].reshape(7, 5)
            noisy_img = noisy_data[idx].reshape(7, 5)
            recon_img = reconstructed[idx].reshape(7, 5)
            
            # Create a separator (column of 0.5s or NaN)
            sep = np.ones((7, 1)) * 0.5
            
            combined = np.hstack([noisy_img, sep, recon_img, sep, orig_img])
            
            ax.imshow(combined, cmap='gray', vmin=0, vmax=1)
            ax.axis('off')
            
            if j == 0:
                ax.set_title(f"Noise: {level*100:.0f}%\nNoisy | Recon | Orig", fontsize=10)
                
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved {output_path}")

def plot_pixel_error_vs_noise(dae, clean_data, noise_levels, output_path):
    """
    Line chart with noise level on X axis, average pixel error on Y axis.
    """
    functions = ["salt_and_pepper", "gaussian", "dropout"]
    
    plt.figure(figsize=(10, 6))
    
    for fn_name in functions:
        noise_fn = get_noise_fn(fn_name)
        avg_errors = []
        
        for level in noise_levels:
            noisy_data = noise_fn(clean_data, level)
            reconstructed = dae.reconstruct(noisy_data)
            
            binary_preds = (reconstructed >= 0.5).astype(int)
            errors_per_char = np.sum(np.abs(binary_preds - clean_data), axis=1)
            avg_errors.append(np.mean(errors_per_char))
            
        plt.plot(noise_levels, avg_errors, marker='o', label=fn_name)
        
    plt.axhline(y=config.MAX_PIXEL_ERROR, color='r', linestyle='--', label='Threshold')
    plt.xlabel('Noise Level')
    plt.ylabel('Average Pixel Error')
    plt.title('Denoising Performance vs Noise Level')
    plt.legend()
    plt.grid(True)
    
    plt.savefig(output_path)
    plt.close()
    print(f"Saved {output_path}")

def plot_reconstruction_quality(dae, clean_data, noise_fn_name, noise_level, output_path):
    """
    For a fixed noise level, show all 32 characters as a 4x8 grid.
    Top half = noisy input, bottom half = reconstruction.
    """
    noise_fn = get_noise_fn(noise_fn_name)
    noisy_data = noise_fn(clean_data, noise_level)
    reconstructed = dae.reconstruct(noisy_data)
    
    fig, axes = plt.subplots(4, 8, figsize=(16, 12))
    axes = axes.flatten()
    
    for i in range(32):
        ax = axes[i]
        
        noisy_img = noisy_data[i].reshape(7, 5)
        recon_img = reconstructed[i].reshape(7, 5)
        
        # Combine vertically: noisy on top, reconstructed on bottom
        sep = np.ones((1, 5)) * 0.5
        combined = np.vstack([noisy_img, sep, recon_img])
        
        ax.imshow(combined, cmap='gray', vmin=0, vmax=1)
        ax.axis('off')
        ax.set_title(f"Char {i}")
        
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved {output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--noise-level", type=float, default=0.20, help="Noise level for specific plots")
    args = parser.parse_args()
    
    clean_data = load_font(os.path.abspath(os.path.join(os.path.dirname(__file__), '../font.h')))
    
    dae = DAE(config.ENCODER_LAYERS, config.DECODER_LAYERS)
    weights_path = os.path.join(os.path.dirname(__file__), '../outputs/dae_weights.npz')
    
    if os.path.exists(weights_path):
        data = np.load(weights_path, allow_pickle=True)
        dae.params = list(data['params'])
    else:
        print("Warning: No weights found. Visualizing with randomly initialized weights.")
        
    out_dir = os.path.join(os.path.dirname(__file__), '../outputs')
    os.makedirs(out_dir, exist_ok=True)
    
    plot_noise_levels_grid(dae, clean_data, config.NOISE_FUNCTION, config.NOISE_LEVELS, 
                           os.path.join(out_dir, 'noise_levels_grid.png'))
                           
    plot_pixel_error_vs_noise(dae, clean_data, config.NOISE_LEVELS, 
                              os.path.join(out_dir, 'pixel_error_vs_noise.png'))
                              
    plot_reconstruction_quality(dae, clean_data, config.NOISE_FUNCTION, args.noise_level, 
                                os.path.join(out_dir, 'reconstruction_quality.png'))

if __name__ == "__main__":
    main()

"""
visualize.py

Generates visualizations for the autoencoder:
1. Latent space scatter plot
2. Original vs. Reconstructed grid
3. New character generation from latent space
Also provides a function to test the pixel error.
"""
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from src.ej1.config import *
from src.ej1.font_loader import load_font
from src.ej1.autoencoder import Autoencoder

def load_trained_model():
    '''
    Initializes model and loads saved weights.
    '''
    model = Autoencoder(
        encoder_layers=ENCODER_LAYERS,
        decoder_layers=DECODER_LAYERS,
        hidden_act=HIDDEN_ACTIVATION,
        out_act=OUTPUT_ACTIVATION
    )
    
    try:
        data = np.load("outputs/best_model.npz")
        # Arrays are named 'arr_0', 'arr_1', etc.
        keys = [f"arr_{i}" for i in range(len(model.params))]
        for i, key in enumerate(keys):
            model.params[i] = data[key]
    except FileNotFoundError:
        print("Error: best_model.npz not found. Train the model first.")
        sys.exit(1)
        
    return model

def check_pixel_errors():
    '''
    Evaluates the model on the 32 characters and prints the pixel error per character.
    '''
    X = load_font("font.h")
    model = load_trained_model()
    
    predictions = model.forward(X)
    
    # Binarize predictions
    binary_preds = (predictions >= 0.5).astype(np.float32)
    
    all_passed = True
    print("--- Pixel Error Test ---")
    for i in range(32):
        errors = np.sum(np.abs(X[i] - binary_preds[i]))
        status = "✓" if errors <= MAX_PIXEL_ERROR else "✗"
        if errors > MAX_PIXEL_ERROR:
            all_passed = False
        print(f"Char {i:2d}: {int(errors):2d} errors {status}")
        
    if all_passed:
        print(f"\nAll 32 characters within threshold (≤{MAX_PIXEL_ERROR} pixel error). ✅")
    else:
        print(f"\nSome characters exceeded threshold (>{MAX_PIXEL_ERROR} pixel error). ❌")
        # Exit with error code if failed
        sys.exit(1)

def plot_reconstructions():
    '''
    Generates reconstructions.png showing original vs reconstructed for all 32 chars.
    '''
    X = load_font("font.h")
    model = load_trained_model()
    predictions = model.forward(X)
    binary_preds = (predictions >= 0.5).astype(np.float32)
    
    fig, axes = plt.subplots(4, 16, figsize=(20, 6))
    fig.suptitle("Original (Top) vs Reconstructed (Bottom)", fontsize=16)
    
    for i in range(32):
        # Original
        ax_orig = axes[(i // 8) * 2, i % 8] if i < 16 else axes[2, i % 16]
        # if i >= 16 we map to row 2
        row_orig = (i // 16) * 2
        col_orig = i % 16
        
        ax_orig = axes[row_orig, col_orig]
        ax_orig.imshow(X[i].reshape(7, 5), cmap="Greys")
        ax_orig.axis("off")
        if i == 0: ax_orig.set_title("Orig")
        
        # Reconstructed
        ax_recon = axes[row_orig + 1, col_orig]
        ax_recon.imshow(binary_preds[i].reshape(7, 5), cmap="Greys")
        ax_recon.axis("off")
        if i == 0: ax_recon.set_title("Recon")
        
    plt.tight_layout()
    os.makedirs("outputs", exist_ok=True)
    plt.savefig("outputs/reconstructions.png")
    print("Saved: outputs/reconstructions.png")
    plt.close()

def plot_latent_space():
    '''
    Generates latent_space.png
    '''
    X = load_font("font.h")
    model = load_trained_model()
    
    latent = model.encode(X)
    
    plt.figure(figsize=(8, 8))
    plt.scatter(latent[:, 0], latent[:, 1], c='blue', marker='o')
    
    # Label each point
    # Mapping font.h index to standard ASCII chars:
    # 0x60 is '`', 0x61 is 'a', etc...
    labels = [chr(0x60 + i) for i in range(32)]
    # 0x7f is DEL, let's just use index if it's not printable
    
    for i in range(32):
        plt.annotate(f"{i}:{labels[i]}", (latent[i, 0], latent[i, 1]), 
                     fontsize=9, alpha=0.7, xytext=(5, 5), textcoords='offset points')
                     
    plt.title("2D Latent Space")
    plt.xlabel("Latent Variable 1")
    plt.ylabel("Latent Variable 2")
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.savefig("outputs/latent_space.png")
    print("Saved: outputs/latent_space.png")
    plt.close()

def generate_new_character():
    '''
    Generates a new character from a point in latent space not corresponding to any training sample.
    '''
    X = load_font("font.h")
    model = load_trained_model()
    
    latent = model.encode(X)
    
    # Find a bounding box for the latent space
    min_x, max_x = np.min(latent[:, 0]), np.max(latent[:, 0])
    min_y, max_y = np.min(latent[:, 1]), np.max(latent[:, 1])
    
    # Pick a point in the middle or offset from mean
    mean_x, mean_y = np.mean(latent[:, 0]), np.mean(latent[:, 1])
    
    # Pick a point that is halfway between mean and max
    new_point = np.array([[mean_x + (max_x - mean_x) * 0.5, mean_y + (max_y - mean_y) * 0.5]])
    
    prediction = model.decode(new_point)
    binary_pred = (prediction >= 0.5).astype(np.float32)
    
    plt.figure(figsize=(3, 4))
    plt.imshow(binary_pred.reshape((7, 5)), cmap="Greys")
    plt.title(f"New Character\nLatent: ({new_point[0,0]:.2f}, {new_point[0,1]:.2f})")
    plt.axis("off")
    
    plt.savefig("outputs/new_character.png")
    print("Saved: outputs/new_character.png")
    plt.close()
    
    # Explanation:
    # This is a "new character" because the point we sampled (new_point)
    # does not match any of the latent points of the 32 training characters.
    # The decoder maps this novel latent coordinate to a 35-dimensional pattern
    # that is a visual interpolation/extrapolation of the learned font features.

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        check_pixel_errors()
    else:
        plot_latent_space()
        plot_reconstructions()
        generate_new_character()

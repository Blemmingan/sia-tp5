"""
visualize_vae.py

Visualization tools for the Variational Autoencoder.
Generates plots for latent space, reconstructions, latent traversal, generated samples, and training curves.
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Add sys.path for activations
_current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(_current_dir, '../../ej1/src')))
sys.path.append(os.path.abspath(os.path.join(_current_dir, '../../src')))

try:
    from activations import get_activation
except ImportError:
    pass

from dataset import load_emoji_dataset
from vae import VAE
import config_vae as cfg

def load_model(outputs_dir):
    weights_path = os.path.join(outputs_dir, 'vae_weights.npz')
    if not os.path.exists(weights_path):
        print(f"Error: Weights file not found at {weights_path}. Train the model first.")
        sys.exit(1)
        
    hidden_activation = get_activation(cfg.HIDDEN_ACTIVATION)
    output_activation = get_activation(cfg.OUTPUT_ACTIVATION)
    
    vae = VAE(
        encoder_layers=cfg.ENCODER_LAYERS,
        latent_dim=cfg.LATENT_DIM,
        decoder_layers=cfg.DECODER_LAYERS,
        activation=hidden_activation,
        output_activation=output_activation
    )
    vae.load(weights_path)
    return vae

def plot_latent_space(vae, x, labels, outputs_dir):
    """
    Comment: We plot mu instead of sampled z because mu represents the deterministic 
    encoding (the mean of the distribution) for each pattern, providing a stable 
    point to visualize. Sampling z would introduce noise, clustering around mu.
    """
    if vae.latent_dim != 2:
        print("Latent dimension is not 2, skipping 2D latent space plot.")
        return
        
    mu, log_var = vae.encode(x)
    
    plt.figure(figsize=(10, 8))
    
    categories = {
        "Faces": ["smiley", "sad face", "surprised face", "neutral face"],
        "Shapes": ["heart", "star", "diamond", "circle"],
        "Arrows": ["arrow up", "arrow down", "arrow left", "arrow right"],
        "Symbols": ["plus", "cross", "checkmark", "moon crescent"]
    }
    
    colors = {"Faces": "red", "Shapes": "blue", "Arrows": "green", "Symbols": "purple"}
    markers = {"Faces": "o", "Shapes": "s", "Arrows": "^", "Symbols": "D"}
    
    for cat_name, cat_labels in categories.items():
        cat_mu = []
        for i, lbl in enumerate(labels):
            if lbl in cat_labels:
                cat_mu.append(mu[i])
                plt.annotate(lbl, (mu[i, 0], mu[i, 1]), xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        if cat_mu:
            cat_mu = np.array(cat_mu)
            plt.scatter(cat_mu[:, 0], cat_mu[:, 1], c=colors[cat_name], marker=markers[cat_name], label=cat_name, s=100)
            
    plt.title("Latent Space (μ)")
    plt.xlabel("Latent Dimension 1")
    plt.ylabel("Latent Dimension 2")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    out_path = os.path.join(outputs_dir, "latent_space.png")
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    print(f"Saved {out_path}")

def plot_reconstructions(vae, x, labels, outputs_dir):
    x_hat, _, _, _ = vae.forward(x)
    
    n_patterns = len(labels)
    fig, axes = plt.subplots(4, 8, figsize=(16, 8))
    fig.suptitle("Original vs Reconstruction", fontsize=16)
    
    for i in range(n_patterns):
        row = i // 4
        col = (i % 4) * 2
        
        ax_orig = axes[row, col]
        ax_recon = axes[row, col + 1]
        
        ax_orig.imshow(x[i].reshape(7, 7), cmap='Greys', interpolation='nearest')
        ax_orig.set_title(f"Orig: {labels[i]}", fontsize=9)
        ax_orig.axis('off')
        
        ax_recon.imshow(x_hat[i].reshape(7, 7), cmap='Greys', interpolation='nearest')
        ax_recon.set_title("Recon", fontsize=9)
        ax_recon.axis('off')
        
    plt.tight_layout()
    out_path = os.path.join(outputs_dir, "reconstructions.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved {out_path}")

def plot_latent_traversal(vae, outputs_dir):
    if vae.latent_dim != 2:
        print("Latent dimension is not 2, skipping traversal plot.")
        return
        
    steps = cfg.LATENT_TRAVERSAL_STEPS
    range_val = cfg.LATENT_TRAVERSAL_RANGE
    traversal_vals = np.linspace(-range_val, range_val, steps)
    
    fig, axes = plt.subplots(2, steps, figsize=(steps * 2, 4))
    fig.suptitle("Latent Traversal", fontsize=16)
    
    # Traversal along dim 0
    for i, val in enumerate(traversal_vals):
        z = np.array([[val, 0.0]])
        x_hat = vae.decode(z)[0]
        ax = axes[0, i]
        ax.imshow(x_hat.reshape(7, 7), cmap='Greys', interpolation='nearest')
        ax.set_title(f"z0={val:.1f}")
        ax.axis('off')
        
    # Traversal along dim 1
    for i, val in enumerate(traversal_vals):
        z = np.array([[0.0, val]])
        x_hat = vae.decode(z)[0]
        ax = axes[1, i]
        ax.imshow(x_hat.reshape(7, 7), cmap='Greys', interpolation='nearest')
        ax.set_title(f"z1={val:.1f}")
        ax.axis('off')
        
    plt.tight_layout()
    out_path = os.path.join(outputs_dir, "latent_traversal.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved {out_path}")

def plot_generated_samples(vae, outputs_dir):
    n_samples = 16
    samples = vae.generate(n_samples=n_samples)
    
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    fig.suptitle("Generated Samples (z ~ N(0,I))", fontsize=16)
    
    for i, ax in enumerate(axes.flatten()):
        ax.imshow(samples[i].reshape(7, 7), cmap='Greys', interpolation='nearest')
        ax.axis('off')
        
    plt.tight_layout()
    out_path = os.path.join(outputs_dir, "generated_samples.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved {out_path}")

def plot_elbo_training(outputs_dir):
    log_file = os.path.join(outputs_dir, 'training_log.csv')
    if not os.path.exists(log_file):
        print("Training log not found.")
        return
        
    data = np.genfromtxt(log_file, delimiter=',', skip_header=1)
    if len(data) == 0:
        return
        
    epochs = data[:, 0]
    total_loss = data[:, 1]
    recon_loss = data[:, 2]
    kl_loss = data[:, 3]
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, total_loss, label='Total ELBO Loss', linewidth=2)
    plt.plot(epochs, recon_loss, label='Reconstruction Term', linestyle='--')
    plt.plot(epochs, kl_loss, label='KL Term', linestyle=':')
    
    plt.title("VAE Training Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    
    out_path = os.path.join(outputs_dir, "elbo_training.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved {out_path}")

def generate_ascii(vae, n):
    samples = vae.generate(n_samples=n)
    print(f"\n--- Generating {n} samples ---")
    for i in range(n):
        print(f"Sample {i+1}:")
        grid = samples[i].reshape((7, 7))
        for row in grid:
            line = "".join("█" if val > 0.5 else " " for val in row)
            print(f"|{line}|")
        print()

def main():
    outputs_dir = os.path.join(os.path.dirname(_current_dir), 'outputs')
    os.makedirs(outputs_dir, exist_ok=True)
    
    # Check for CLI arguments
    if len(sys.argv) > 1 and sys.argv[1] == '--generate':
        n = int(sys.argv[2]) if len(sys.argv) > 2 else cfg.N_GENERATE
        vae = load_model(outputs_dir)
        generate_ascii(vae, n)
        return
        
    x, labels = load_emoji_dataset()
    vae = load_model(outputs_dir)
    
    plot_latent_space(vae, x, labels, outputs_dir)
    plot_reconstructions(vae, x, labels, outputs_dir)
    plot_latent_traversal(vae, outputs_dir)
    plot_generated_samples(vae, outputs_dir)
    plot_elbo_training(outputs_dir)

if __name__ == '__main__':
    main()

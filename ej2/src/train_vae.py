"""
train_vae.py

Training script for the Variational Autoencoder.
"""
import os
import sys
import numpy as np

# Add both possible locations for activations.py and optimizers.py
_current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(_current_dir, '../../ej1/src')))
sys.path.append(os.path.abspath(os.path.join(_current_dir, '../../src')))

try:
    from activations import get_activation
    from optimizers import get_optimizer
except ImportError:
    print("Error: Could not import activations or optimizers.")
    sys.exit(1)

from dataset import load_emoji_dataset
from vae import VAE
from losses import elbo_loss
import config_vae as cfg

def main():
    # Setup outputs dir
    outputs_dir = os.path.join(os.path.dirname(_current_dir), 'outputs')
    os.makedirs(outputs_dir, exist_ok=True)
    
    # 1. Load dataset
    x, labels = load_emoji_dataset()
    
    # 2. Instantiate VAE
    hidden_activation = get_activation(cfg.HIDDEN_ACTIVATION)
    output_activation = get_activation(cfg.OUTPUT_ACTIVATION)
    
    vae = VAE(
        encoder_layers=cfg.ENCODER_LAYERS,
        latent_dim=cfg.LATENT_DIM,
        decoder_layers=cfg.DECODER_LAYERS,
        activation=hidden_activation,
        output_activation=output_activation
    )
    
    optimizer = get_optimizer(cfg.OPTIMIZER, lr=cfg.LEARNING_RATE)
    
    # Check if CLI overrides beta
    beta = cfg.BETA
    for arg in sys.argv[1:]:
        if arg.startswith('--beta='):
            beta = float(arg.split('=')[1])
        if arg.startswith('--latent_dim='):
            # This would require re-instantiating VAE if we actually support it
            pass
            
    print(f"Training VAE (Latent Dim: {vae.latent_dim}, Beta: {beta}, Loss: {cfg.LOSS})")
    
    log_data = []
    
    # 3. Training Loop
    for epoch in range(1, cfg.EPOCHS + 1):
        # Forward pass
        x_hat, mu, log_var, z = vae.forward(x)
        
        # Compute ELBO
        total_loss, recon, kl = elbo_loss(x, x_hat, mu, log_var, beta=beta, loss_type=cfg.LOSS)
        
        # Backward pass
        grads = vae.backward(x, x_hat, mu, log_var, z, vae.cache['epsilon'], loss_type=cfg.LOSS, beta=beta)
        
        # Optimizer step
        params_list, grads_list = vae.get_params_and_grads_lists(grads)
        optimizer.update(params_list, grads_list)
        
        # Logging
        if epoch % 500 == 0 or epoch == 1:
            print(f"Epoch {epoch:04d} | Total: {total_loss:.4f} | Recon: {recon:.4f} | KL: {kl:.4f}")
            
        if epoch % 100 == 0:
            log_data.append([epoch, total_loss, recon, kl])
            
    # Save training log
    log_file = os.path.join(outputs_dir, 'training_log.csv')
    with open(log_file, 'w') as f:
        f.write("epoch,total_loss,recon_loss,kl_loss\n")
        for row in log_data:
            f.write(f"{row[0]},{row[1]:.6f},{row[2]:.6f},{row[3]:.6f}\n")
            
    # Save weights
    weights_file = os.path.join(outputs_dir, 'vae_weights.npz')
    vae.save(weights_file)
    print(f"Saved weights to {weights_file}")
    
    # After training: per-pattern pixel error table
    x_hat_final, _, _, _ = vae.forward(x)
    x_hat_bin = (x_hat_final > 0.5).astype(int)
    x_bin = x.astype(int)
    
    errors = np.sum(x_hat_bin != x_bin, axis=1)
    
    print("\n--- Per-Pattern Pixel Error ---")
    for i, lbl in enumerate(labels):
        print(f"{lbl:15s}: {errors[i]:2d} pixels wrong")
        
    print(f"Total Errors: {np.sum(errors)} / {x.shape[0]*x.shape[1]}")
    
    # Print generated sample
    print("\n--- Generated Sample (z ~ N(0, I)) ---")
    new_sample = vae.generate(n_samples=1)[0]
    grid = new_sample.reshape((7, 7))
    for row in grid:
        line = "".join("█" if val > 0.5 else " " for val in row)
        print(f"|{line}|")
        
    # Write summary files for sweeps if requested
    if '--sweep' in sys.argv:
        # e.g. --sweep beta_sweep.csv
        sweep_idx = sys.argv.index('--sweep')
        if sweep_idx + 1 < len(sys.argv):
            out_csv = sys.argv[sweep_idx + 1]
            sweep_path = os.path.join(outputs_dir, out_csv)
            file_exists = os.path.isfile(sweep_path)
            with open(sweep_path, 'a') as f:
                if out_csv == 'beta_sweep.csv':
                    if not file_exists:
                        f.write("beta,final_recon_loss,final_kl,pixel_error\n")
                    f.write(f"{beta},{recon:.6f},{kl:.6f},{np.sum(errors)}\n")
                elif out_csv == 'latent_sweep.csv':
                    if not file_exists:
                        f.write("latent_dim,final_loss,pixel_error\n")
                    f.write(f"{vae.latent_dim},{total_loss:.6f},{np.sum(errors)}\n")

if __name__ == '__main__':
    main()

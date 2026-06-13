"""
config_vae.py

Configuration parameters for the Variational Autoencoder.
"""

# Architecture
ENCODER_LAYERS = [49, 32, 16]     # Excludes latent — handled separately
LATENT_DIM = 2                    # 2 for visualization; try 4, 8 for better reconstruction
DECODER_LAYERS = [16, 32, 49]     # Excludes latent input — handled separately

# Training
LEARNING_RATE = 0.001
EPOCHS = 10000
OPTIMIZER = "adam"
LOSS = "bce"                      # Reconstruction loss type: "bce" | "mse"
BETA = 1.0                        # KL weight: 1.0 = standard VAE, >1 = beta-VAE

# Activations
HIDDEN_ACTIVATION = "tanh"
OUTPUT_ACTIVATION = "sigmoid"

# Generation
N_GENERATE = 16                   # Samples to generate for visualize --generate
LATENT_TRAVERSAL_STEPS = 10      # Steps for latent space traversal plot
LATENT_TRAVERSAL_RANGE = 3.0     # Range: -3.0 to +3.0

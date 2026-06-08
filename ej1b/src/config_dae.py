"""
config_dae.py

Configuration constants for the Denoising Autoencoder (DAE).
Includes architecture, training hyperparameters, and noise settings.
"""

# Architecture
# Uses a wider latent representation (8D) compared to the 2D visualization bottleneck
# in ej1a to provide sufficient capacity for denoising the characters.
ENCODER_LAYERS = [35, 32, 16, 8]
DECODER_LAYERS = [8, 16, 32, 35]

# Training
LEARNING_RATE = 0.001
EPOCHS = 5000
OPTIMIZER = "adam"
LOSS = "bce"

# Noise
NOISE_FUNCTION = "salt_and_pepper"   # "salt_and_pepper" | "gaussian" | "dropout"
NOISE_LEVELS = [0.05, 0.10, 0.20, 0.30, 0.40]
TRAIN_NOISE_LEVEL = 0.10             # Noise level used during training

# Thresholds
MAX_PIXEL_ERROR = 1

"""
config.py

Contains all hyperparameters and configuration settings for the autoencoder project.
"""

# Architecture
# Input must be 35 (5x7), latent must be 2, output must be 35.
ENCODER_LAYERS = [35, 16, 8, 2]   # Input -> hidden -> latent
DECODER_LAYERS = [2, 8, 16, 35]   # Latent -> hidden -> output

# Training hyperparameters
LEARNING_RATE = 0.01
EPOCHS = 10000
BATCH_SIZE = 32                    # Full batch for all 32 characters by default
OPTIMIZER = "adam"                 # "sgd" | "momentum" | "adam"
LOSS = "bce"                       # "bce" | "mse"
BETA_1 = 0.9                       # Adam parameter
BETA_2 = 0.999                     # Adam parameter
MOMENTUM_COEF = 0.9                # Momentum optimizer parameter
GRAD_CLIP_VAL = 1.0                # Gradient clipping to avoid explosions

# Thresholds
MAX_PIXEL_ERROR = 1                # Max incorrect pixels per character

# Activations
HIDDEN_ACTIVATION = "tanh"         # Encoder/decoder hidden layers ("sigmoid", "tanh", "relu", "leaky_relu", "linear")
OUTPUT_ACTIVATION = "sigmoid"      # Output layer (since we reconstruct binary pixels {0, 1})

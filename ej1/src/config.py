"""
config.py

Contains all hyperparameters and configuration settings for the autoencoder project.
"""
import argparse

parser = argparse.ArgumentParser(description="Autoencoder Config", add_help=False)
parser.add_argument("--encoder", type=str, default="35,16,8,2")
parser.add_argument("--decoder", type=str, default="2,8,16,35")
parser.add_argument("--lr", type=float, default=0.01)
parser.add_argument("--epochs", type=int, default=30000)
parser.add_argument("--optimizer", type=str, default="adam")
parser.add_argument("--loss", type=str, default="bce")
parser.add_argument("--hidden_act", type=str, default="tanh")
parser.add_argument("--out_act", type=str, default="sigmoid")

args, _ = parser.parse_known_args()

# Architecture
ENCODER_LAYERS = [int(x) for x in args.encoder.split(',')]
DECODER_LAYERS = [int(x) for x in args.decoder.split(',')]

# Training hyperparameters
LEARNING_RATE = args.lr
EPOCHS = args.epochs
BATCH_SIZE = 32                    # Full batch for all 32 characters by default
OPTIMIZER = args.optimizer                 # "sgd" | "momentum" | "adam"
LOSS = args.loss                       # "bce" | "mse"
BETA_1 = 0.9                       # Adam parameter
BETA_2 = 0.999                     # Adam parameter
MOMENTUM_COEF = 0.9                # Momentum optimizer parameter
GRAD_CLIP_VAL = 1.0                # Gradient clipping to avoid explosions

# Thresholds
MAX_PIXEL_ERROR = 1                # Max incorrect pixels per character

# Activations
HIDDEN_ACTIVATION = args.hidden_act         # Encoder/decoder hidden layers ("sigmoid", "tanh", "relu", "leaky_relu", "linear")
OUTPUT_ACTIVATION = args.out_act      # Output layer (since we reconstruct binary pixels {0, 1})


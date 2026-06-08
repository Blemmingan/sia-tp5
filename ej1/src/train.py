"""
train.py

Training loop for the autoencoder.
Loads data, sets up network and optimizer, trains for N epochs,
and saves the best model.
"""
import numpy as np
import csv
import os
from src.config import *
from src.font_loader import load_font
from src.autoencoder import Autoencoder
from src.optimizers import get_optimizer

def train():
    print("Loading data...")
    X = load_font("font.h")
    # For autoencoder, input is the target
    Y = X
    
    print(f"Dataset shape: {X.shape}")
    
    # Initialize model
    model = Autoencoder(
        encoder_layers=ENCODER_LAYERS,
        decoder_layers=DECODER_LAYERS,
        hidden_act=HIDDEN_ACTIVATION,
        out_act=OUTPUT_ACTIVATION
    )
    
    # Initialize optimizer
    opt = get_optimizer(
        OPTIMIZER, 
        lr=LEARNING_RATE, 
        beta1=BETA_1, 
        beta2=BETA_2, 
        momentum=MOMENTUM_COEF
    )
    
    best_loss = float('inf')
    best_params = None
    
    history = []
    
    print(f"Starting training for {EPOCHS} epochs using {OPTIMIZER.upper()} optimizer and {LOSS.upper()} loss...")
    
    for epoch in range(EPOCHS):
        # Forward pass
        model.forward(X)
        
        # Backward pass
        grads, loss = model.backward(X, Y, loss_type=LOSS, grad_clip_val=GRAD_CLIP_VAL)
        
        # Update weights
        opt.update(model.params, grads)
        
        history.append(loss)
        
        if epoch % 100 == 0 or epoch == EPOCHS - 1:
            print(f"Epoch {epoch:<5} | Loss: {loss:.6f}")
            
        if loss < best_loss:
            best_loss = loss
            # Copy parameters to save the best model
            best_params = [np.copy(p) for p in model.params]
            
    print("Training complete.")
    
    # Save best model
    os.makedirs("outputs", exist_ok=True)
    
    model_path = "outputs/best_model.npz"
    np.savez(model_path, *best_params)
    print(f"Best model saved to {model_path} with loss: {best_loss:.6f}")
    
    # Save training log
    log_path = "outputs/training_log.csv"
    with open(log_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Epoch", "Loss"])
        for i, l in enumerate(history):
            writer.writerow([i, l])
    print(f"Training log saved to {log_path}")

if __name__ == "__main__":
    train()

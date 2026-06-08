"""
dae.py

Implements a Denoising Autoencoder (DAE) from scratch using NumPy.
Reuses the core forward/backward propagation logic but trains by
injecting noise into the inputs while computing loss against the clean originals.
"""
import sys
import os
import numpy as np

# Ensure parent directory is in path to import from ej1.src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from ej1.src.activations import get_activation

class DAE:
    def __init__(self, encoder_layers, decoder_layers, hidden_act="tanh", out_act="sigmoid"):
        '''
        Initializes the denoising autoencoder architecture.
        
        Why an 8D bottleneck? 
        The DAE needs sufficient latent capacity to map highly noisy input variations 
        back to the exact clean original. While a 2D space forces a strict continuous 
        manifold (useful for visualization), an 8D space provides the redundancy needed 
        to store robust representations that ignore noise perturbations.
        '''
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        
        self.hidden_activation = get_activation(hidden_act)
        self.output_activation = get_activation(out_act)
        
        self.params = []
        
        # Combine layers. decoder_layers[0] is the latent layer size
        self.architecture = encoder_layers + decoder_layers[1:]
        self.num_layers = len(self.architecture) - 1
        self.latent_idx = len(encoder_layers) - 1
        
        self._initialize_weights()
        
    def _initialize_weights(self):
        '''Xavier/Glorot uniform initialization for weights, zeros for biases.'''
        for i in range(self.num_layers):
            fan_in = self.architecture[i]
            fan_out = self.architecture[i+1]
            
            limit = np.sqrt(6.0 / (fan_in + fan_out))
            W = np.random.uniform(-limit, limit, size=(fan_in, fan_out))
            b = np.zeros((1, fan_out))
            
            self.params.append(W)
            self.params.append(b)
            
    def forward(self, X):
        '''
        Forward pass.
        '''
        self.A = [X]
        self.Z = []
        
        current_A = X
        
        for i in range(self.num_layers):
            W = self.params[2*i]
            b = self.params[2*i + 1]
            
            # Z = A * W + b
            Z = np.dot(current_A, W) + b
            self.Z.append(Z)
            
            # Apply activation
            if i == self.num_layers - 1:
                current_A = self.output_activation.forward(Z)
            else:
                current_A = self.hidden_activation.forward(Z)
                
            self.A.append(current_A)
            
        return current_A
        
    def backward(self, Y_clean, loss_type="bce", grad_clip_val=1.0):
        '''
        Backward pass to compute gradients against the CLEAN targets.
        
        Args:
            Y_clean (np.ndarray): Original, uncorrupted target data.
            loss_type (str): "bce" or "mse"
            grad_clip_val (float): Gradient clipping max norm.
        '''
        m = Y_clean.shape[0]
        predictions = self.A[-1]
        
        # 1. Loss computation against CLEAN target
        if loss_type == "bce":
            eps = 1e-15
            preds_clipped = np.clip(predictions, eps, 1.0 - eps)
            loss = -np.mean(Y_clean * np.log(preds_clipped) + (1 - Y_clean) * np.log(1 - preds_clipped))
            
            # Derivative simplified for BCE + Sigmoid
            if self.output_activation.__name__ == 'Sigmoid':
                delta = (predictions - Y_clean) / m
            else:
                d_loss_d_A = ((1 - Y_clean) / (1 - preds_clipped) - Y_clean / preds_clipped) / m
                delta = d_loss_d_A * self.output_activation.backward(self.Z[-1])
                
        elif loss_type == "mse":
            loss = np.mean((predictions - Y_clean) ** 2)
            d_loss_d_A = 2 * (predictions - Y_clean) / (m * Y_clean.shape[1])
            delta = d_loss_d_A * self.output_activation.backward(self.Z[-1])
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
            
        grads = [None] * len(self.params)
        
        # 2. Backpropagate
        for i in reversed(range(self.num_layers)):
            A_prev = self.A[i]
            dW = np.dot(A_prev.T, delta)
            db = np.sum(delta, axis=0, keepdims=True)
            
            grads[2*i] = dW
            grads[2*i + 1] = db
            
            if i > 0:
                W_current = self.params[2*i]
                delta = np.dot(delta, W_current.T) * self.hidden_activation.backward(self.Z[i-1])
                
        # 3. Gradient Clipping
        if grad_clip_val is not None:
            for j in range(len(grads)):
                grads[j] = np.clip(grads[j], -grad_clip_val, grad_clip_val)
                
        return grads, loss

    def reconstruct(self, noisy_input):
        '''
        Helper to return clean output from noisy input.
        '''
        return self.forward(noisy_input)

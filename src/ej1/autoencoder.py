"""
autoencoder.py

Implements a basic Autoencoder from scratch using NumPy.
Handles forward propagation, backpropagation (chain rule), and loss calculations.
"""
import numpy as np
from src.ej1.activations import get_activation

class Autoencoder:
    def __init__(self, encoder_layers, decoder_layers, hidden_act="tanh", out_act="sigmoid"):
        '''
        Initializes the autoencoder layers, weights, and activations.
        '''
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        
        self.hidden_activation = get_activation(hidden_act)
        self.output_activation = get_activation(out_act)
        
        # We will store parameters in a list: [W1, b1, W2, b2, ...]
        self.params = []
        
        # Combine layers to initialize weights in one loop
        # The latent layer is the end of encoder / start of decoder.
        # However, decoder_layers[0] is the latent layer size, so we don't repeat it.
        # e.g., enc=[35,16,8,2], dec=[2,8,16,35] -> full architecture: [35,16,8,2,8,16,35]
        self.architecture = encoder_layers + decoder_layers[1:]
        self.num_layers = len(self.architecture) - 1
        self.latent_idx = len(encoder_layers) - 1 # index of the latent layer in the forward pass
        
        self._initialize_weights()
        
    def _initialize_weights(self):
        '''
        Initializes weights using Xavier/Glorot uniform initialization.
        Biases are initialized to zeros.
        '''
        for i in range(self.num_layers):
            fan_in = self.architecture[i]
            fan_out = self.architecture[i+1]
            
            # Xavier uniform limit
            limit = np.sqrt(6.0 / (fan_in + fan_out))
            W = np.random.uniform(-limit, limit, size=(fan_in, fan_out))
            b = np.zeros((1, fan_out))
            
            self.params.append(W)
            self.params.append(b)
            
    def forward(self, X):
        '''
        Forward pass.
        
        Args:
            X (np.ndarray): Input data of shape (batch_size, input_dim)
            
        Returns:
            np.ndarray: Output predictions of shape (batch_size, output_dim)
        '''
        self.A = [X] # Store activations for backprop
        self.Z = []  # Store pre-activations for backprop
        
        current_A = X
        
        for i in range(self.num_layers):
            W = self.params[2*i]
            b = self.params[2*i + 1]
            
            # Z = A * W + b
            Z = np.dot(current_A, W) + b
            self.Z.append(Z)
            
            # Apply activation
            if i == self.num_layers - 1:
                # Output layer
                current_A = self.output_activation.forward(Z)
            else:
                # Hidden layers
                # Note: Latent layer also uses hidden activation or linear, 
                # but standard practice is to use hidden activation or linear.
                # Here we use hidden activation for all intermediate layers.
                current_A = self.hidden_activation.forward(Z)
                
            self.A.append(current_A)
            
        return current_A
        
    def encode(self, X):
        '''
        Returns the latent representation for input X.
        '''
        current_A = X
        # Forward pass up to the latent layer
        for i in range(self.latent_idx):
            W = self.params[2*i]
            b = self.params[2*i + 1]
            Z = np.dot(current_A, W) + b
            current_A = self.hidden_activation.forward(Z)
        return current_A
        
    def decode(self, latent_Z):
        '''
        Returns the reconstruction for latent representation latent_Z.
        '''
        current_A = latent_Z
        # Forward pass from latent layer to output
        for i in range(self.latent_idx, self.num_layers):
            W = self.params[2*i]
            b = self.params[2*i + 1]
            Z = np.dot(current_A, W) + b
            if i == self.num_layers - 1:
                current_A = self.output_activation.forward(Z)
            else:
                current_A = self.hidden_activation.forward(Z)
        return current_A

    def backward(self, X, Y, loss_type="bce", grad_clip_val=1.0):
        '''
        Backward pass to compute gradients.
        
        Args:
            X (np.ndarray): Input data (not strictly needed as it's in self.A[0], but good for clarity)
            Y (np.ndarray): Target data
            loss_type (str): "bce" or "mse"
            grad_clip_val (float): Maximum norm for gradient clipping
            
        Returns:
            list: Gradients for each parameter in self.params [dW1, db1, dW2, db2, ...]
            float: Computed loss
        '''
        m = Y.shape[0] # batch size
        predictions = self.A[-1]
        
        # 1. Compute Loss and initial derivative (delta)
        if loss_type == "bce":
            # Binary Cross Entropy
            # Clip predictions to avoid log(0)
            eps = 1e-15
            preds_clipped = np.clip(predictions, eps, 1.0 - eps)
            loss = -np.mean(Y * np.log(preds_clipped) + (1 - Y) * np.log(1 - preds_clipped))
            
            # dL/dZ for BCE + Sigmoid = (predictions - Y) / m
            # Note: This simplification holds ONLY if output activation is Sigmoid.
            if self.output_activation.__name__ == 'Sigmoid':
                delta = (predictions - Y) / m
            else:
                # Generic fallback if not sigmoid:
                # dL/dA = ( (1-Y)/(1-A) - Y/A ) / m
                d_loss_d_A = ((1 - Y) / (1 - preds_clipped) - Y / preds_clipped) / m
                delta = d_loss_d_A * self.output_activation.backward(self.Z[-1])
                
        elif loss_type == "mse":
            # Mean Squared Error
            loss = np.mean((predictions - Y) ** 2)
            
            # dL/dA = 2 * (predictions - Y) / (m * num_features)
            # Actually standard MSE derivative for sum of squares over m is just:
            d_loss_d_A = 2 * (predictions - Y) / (m * Y.shape[1])
            delta = d_loss_d_A * self.output_activation.backward(self.Z[-1])
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
            
        grads = [None] * len(self.params)
        
        # 2. Backpropagate through layers
        # Iterate backwards
        for i in reversed(range(self.num_layers)):
            # Activations from previous layer
            A_prev = self.A[i]
            
            # dL/dW = A_prev^T @ delta
            dW = np.dot(A_prev.T, delta)
            
            # dL/db = sum of delta over batch
            db = np.sum(delta, axis=0, keepdims=True)
            
            grads[2*i] = dW
            grads[2*i + 1] = db
            
            if i > 0:
                W_current = self.params[2*i]
                # delta for next layer = (delta @ W_current^T) * f'(Z_prev)
                delta = np.dot(delta, W_current.T) * self.hidden_activation.backward(self.Z[i-1])
                
        # 3. Gradient Clipping to avoid exploding gradients
        if grad_clip_val is not None:
            for j in range(len(grads)):
                grads[j] = np.clip(grads[j], -grad_clip_val, grad_clip_val)
                
        return grads, loss

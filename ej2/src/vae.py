"""
vae.py

Variational Autoencoder (VAE) implementation from scratch using NumPy.
Includes forward pass with reparameterization trick and full manual backpropagation.
Reference: Auto-Encoding Variational Bayes (Kingma & Welling, 2013).
"""
import numpy as np

class VAE:
    def __init__(self, encoder_layers, latent_dim, decoder_layers, activation, output_activation):
        """
        Initialize weights using Xavier initialization.
        
        Args:
            encoder_layers (list): e.g. [49, 32, 16] (input, hidden1, hidden2)
            latent_dim (int): size of the latent space (e.g. 2)
            decoder_layers (list): e.g. [16, 32, 49] (hidden1, hidden2, output)
            activation: Activation function class for hidden layers
            output_activation: Activation function class for output layer
        """
        self.latent_dim = latent_dim
        self.activation = activation
        self.output_activation = output_activation
        
        self.params = {}
        
        # Encoder weights
        # Shapes: W_enc_1: (49, 32), b_enc_1: (1, 32), etc.
        self.n_enc_layers = len(encoder_layers) - 1
        for i in range(self.n_enc_layers):
            in_dim = encoder_layers[i]
            out_dim = encoder_layers[i+1]
            # Xavier init
            limit = np.sqrt(6 / (in_dim + out_dim))
            self.params[f'W_enc_{i}'] = np.random.uniform(-limit, limit, (in_dim, out_dim))
            self.params[f'b_enc_{i}'] = np.zeros((1, out_dim))
            
        # Latent projection weights
        # mu layer: (encoder_last, latent_dim)
        last_enc_dim = encoder_layers[-1]
        limit = np.sqrt(6 / (last_enc_dim + latent_dim))
        self.params['W_mu'] = np.random.uniform(-limit, limit, (last_enc_dim, latent_dim))
        self.params['b_mu'] = np.zeros((1, latent_dim))
        
        # log_var layer: (encoder_last, latent_dim)
        self.params['W_log_var'] = np.random.uniform(-limit, limit, (last_enc_dim, latent_dim))
        self.params['b_log_var'] = np.zeros((1, latent_dim))
        
        # Decoder weights
        # Shapes: W_dec_0: (latent_dim, 16), b_dec_0: (1, 16), etc.
        self.n_dec_layers = len(decoder_layers)
        
        # First decoder layer takes latent_dim as input
        in_dim = latent_dim
        out_dim = decoder_layers[0]
        limit = np.sqrt(6 / (in_dim + out_dim))
        self.params['W_dec_0'] = np.random.uniform(-limit, limit, (in_dim, out_dim))
        self.params['b_dec_0'] = np.zeros((1, out_dim))
        
        for i in range(1, self.n_dec_layers):
            in_dim = decoder_layers[i-1]
            out_dim = decoder_layers[i]
            limit = np.sqrt(6 / (in_dim + out_dim))
            self.params[f'W_dec_{i}'] = np.random.uniform(-limit, limit, (in_dim, out_dim))
            self.params[f'b_dec_{i}'] = np.zeros((1, out_dim))
            
        # Cache for backprop
        self.cache = {}

    def encode(self, x):
        """
        Forward pass through encoder.
        Comment: we output log_var instead of var (variance) for numerical stability.
        Log variance can take any real value (-inf, +inf) which is easier for a 
        neural network to output unconstrained, whereas variance must be strictly positive.
        
        Returns: 
            mu (np.ndarray): shape (batch_size, latent_dim)
            log_var (np.ndarray): shape (batch_size, latent_dim)
        """
        a = x
        self.cache['a_enc_0'] = a
        
        for i in range(self.n_enc_layers):
            z_pre = np.dot(a, self.params[f'W_enc_{i}']) + self.params[f'b_enc_{i}']
            self.cache[f'z_enc_{i}'] = z_pre
            a = self.activation.forward(z_pre)
            self.cache[f'a_enc_{i+1}'] = a
            
        # Compute mu and log_var
        mu = np.dot(a, self.params['W_mu']) + self.params['b_mu']
        log_var = np.dot(a, self.params['W_log_var']) + self.params['b_log_var']
        
        self.cache['mu'] = mu
        self.cache['log_var'] = log_var
        return mu, log_var

    def reparameterize(self, mu, log_var):
        """
        Sample z using the reparameterization trick.
        z = mu + sigma * epsilon, epsilon ~ N(0, I)
        
        Comment: why this trick is needed for backpropagation.
        Random sampling operation z ~ N(mu, sigma^2) is not differentiable.
        By moving the randomness to an independent variable epsilon ~ N(0,I),
        the path from mu and log_var to z becomes a simple differentiable affine transformation.
        This allows gradients to flow backwards through the random node.
        
        Returns: 
            z (np.ndarray), epsilon (np.ndarray)
        """
        std = np.exp(0.5 * log_var)
        epsilon = np.random.randn(*mu.shape)
        z = mu + std * epsilon
        return z, epsilon

    def decode(self, z):
        """
        Forward pass through decoder.
        Returns reconstruction x_hat.
        """
        a = z
        self.cache['a_dec_0'] = a
        
        for i in range(self.n_dec_layers):
            z_pre = np.dot(a, self.params[f'W_dec_{i}']) + self.params[f'b_dec_{i}']
            self.cache[f'z_dec_{i}'] = z_pre
            
            # Last layer uses output_activation
            if i == self.n_dec_layers - 1:
                a = self.output_activation.forward(z_pre)
            else:
                a = self.activation.forward(z_pre)
            self.cache[f'a_dec_{i+1}'] = a
            
        x_hat = a
        return x_hat

    def forward(self, x):
        """
        Full forward pass.
        Returns: x_hat, mu, log_var, z
        """
        mu, log_var = self.encode(x)
        z, epsilon = self.reparameterize(mu, log_var)
        self.cache['epsilon'] = epsilon
        x_hat = self.decode(z)
        return x_hat, mu, log_var, z

    def backward(self, x, x_hat, mu, log_var, z, epsilon, loss_type="bce", beta=1.0):
        """
        Full backpropagation through decoder, sampling, and encoder.
        Returns: dict of all parameter gradients.
        """
        batch_size = x.shape[0]
        grads = {}
        
        # ====================================================
        # 1. dL/d(x_hat): reconstruction loss gradient
        # ====================================================
        # Comment: Gradient of the loss with respect to the output layer
        if loss_type == "bce":
            # Add epsilon for numerical stability
            eps = 1e-12
            x_hat_clip = np.clip(x_hat, eps, 1.0 - eps)
            # Derivative of BCE: ( (1-x)/(1-x_hat) - x/x_hat ) / batch_size
            dl_dxhat = ( (1 - x) / (1.0 - x_hat_clip) - x / x_hat_clip ) / batch_size
        else:
            # Derivative of MSE
            dl_dxhat = 2.0 * (x_hat - x) / (batch_size * x.shape[1])
            
        # Backprop through output activation
        da = dl_dxhat
        dz_pre = da * self.output_activation.backward(self.cache[f'z_dec_{self.n_dec_layers-1}'])
        
        # ====================================================
        # 2. Decoder Backpropagation
        # ====================================================
        for i in reversed(range(self.n_dec_layers)):
            a_prev = self.cache[f'a_dec_{i}']
            
            # Gradients for weights and biases
            grads[f'W_dec_{i}'] = np.dot(a_prev.T, dz_pre)
            grads[f'b_dec_{i}'] = np.sum(dz_pre, axis=0, keepdims=True)
            
            # Gradient w.r.t input of this layer
            da_prev = np.dot(dz_pre, self.params[f'W_dec_{i}'].T)
            
            if i > 0:
                # Backprop through hidden activation
                dz_pre = da_prev * self.activation.backward(self.cache[f'z_dec_{i-1}'])
                
        # Gradient w.r.t latent representation z
        dl_dz = da_prev
        
        # ====================================================
        # 3. Gradient through reparameterization: dz/d(mu), dz/d(log_var)
        # ====================================================
        # z = mu + exp(0.5 * log_var) * epsilon
        # dL/d(mu) from reconstruction path = dL/dz * dz/d(mu)
        # dz/d(mu) = 1, so dL/d(mu)_recon = dL/dz
        dl_dmu_recon = dl_dz
        
        # dL/d(log_var) from reconstruction path = dL/dz * dz/d(log_var)
        # dz/d(log_var) = 0.5 * exp(0.5 * log_var) * epsilon
        dl_dlogvar_recon = dl_dz * 0.5 * np.exp(0.5 * log_var) * epsilon
        
        # ====================================================
        # 4. KL Divergence Gradients
        # ====================================================
        # KL = -0.5 * sum(1 + log_var - mu^2 - exp(log_var))
        # dL/d(mu)_KL = beta * d(KL)/d(mu) = beta * mu / batch_size
        dl_dmu_kl = beta * mu / batch_size
        
        # dL/d(log_var)_KL = beta * d(KL)/d(log_var) = beta * 0.5 * (exp(log_var) - 1) / batch_size
        dl_dlogvar_kl = beta * 0.5 * (np.exp(log_var) - 1.0) / batch_size
        
        # Total gradients for mu and log_var
        # Comment: dL/d(mu): KL + reconstruction path
        dl_dmu = dl_dmu_recon + dl_dmu_kl
        
        # Comment: dL/d(log_var): KL gradient + reconstruction path
        dl_dlogvar = dl_dlogvar_recon + dl_dlogvar_kl
        
        # ====================================================
        # 5. Encoder Backpropagation
        # ====================================================
        a_last_enc = self.cache[f'a_enc_{self.n_enc_layers}']
        
        # Gradients for mu projection layer
        grads['W_mu'] = np.dot(a_last_enc.T, dl_dmu)
        grads['b_mu'] = np.sum(dl_dmu, axis=0, keepdims=True)
        
        # Gradients for log_var projection layer
        grads['W_log_var'] = np.dot(a_last_enc.T, dl_dlogvar)
        grads['b_log_var'] = np.sum(dl_dlogvar, axis=0, keepdims=True)
        
        # Gradient w.r.t output of the last hidden encoder layer
        da_enc_last = np.dot(dl_dmu, self.params['W_mu'].T) + np.dot(dl_dlogvar, self.params['W_log_var'].T)
        
        dz_pre = da_enc_last * self.activation.backward(self.cache[f'z_enc_{self.n_enc_layers-1}'])
        
        for i in reversed(range(self.n_enc_layers)):
            a_prev = self.cache[f'a_enc_{i}']
            
            # Gradients for weights and biases
            grads[f'W_enc_{i}'] = np.dot(a_prev.T, dz_pre)
            grads[f'b_enc_{i}'] = np.sum(dz_pre, axis=0, keepdims=True)
            
            # Gradient w.r.t input of this layer
            da_prev = np.dot(dz_pre, self.params[f'W_enc_{i}'].T)
            
            if i > 0:
                # Backprop through hidden activation
                dz_pre = da_prev * self.activation.backward(self.cache[f'z_enc_{i-1}'])
                
        return grads
        
    def generate(self, z=None, n_samples=1):
        """
        Generate new patterns.
        If z is None, sample z ~ N(0, I).
        Returns: decoded patterns of shape (n_samples, input_dim)
        """
        if z is None:
            z = np.random.randn(n_samples, self.latent_dim)
        return self.decode(z)
        
    def save(self, path):
        """Save model parameters."""
        np.savez(path, **self.params)
        
    def load(self, path):
        """Load model parameters."""
        loaded = np.load(path)
        for key in self.params.keys():
            self.params[key] = loaded[key]

    def get_params_and_grads_lists(self, grads_dict):
        """Helper to align params and grads for the optimizers."""
        params_list = []
        grads_list = []
        
        # Ensure consistent order
        keys = list(self.params.keys())
        for k in keys:
            params_list.append(self.params[k])
            grads_list.append(grads_dict[k])
            
        return params_list, grads_list

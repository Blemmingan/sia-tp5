"""
losses.py

Implements ELBO loss components for the Variational Autoencoder.
"""
import numpy as np

def reconstruction_loss(x, x_hat, loss_type="bce"):
    """
    Binary cross-entropy or MSE between original x and reconstruction x_hat.
    
    Comment: BCE is preferred for binary inputs because it treats each pixel
    as an independent Bernoulli variable, calculating the negative log likelihood.
    
    Args:
        x (np.ndarray): Original input patterns
        x_hat (np.ndarray): Reconstructed patterns
        loss_type (str): "bce" or "mse"
        
    Returns:
        float: the scalar loss over the batch
    """
    if loss_type.lower() == "mse":
        return np.mean((x - x_hat) ** 2)
    elif loss_type.lower() == "bce":
        # Add epsilon to prevent log(0)
        eps = 1e-12
        x_hat = np.clip(x_hat, eps, 1.0 - eps)
        # BCE: -mean(x * log(x_hat) + (1-x) * log(1-x_hat))
        bce = -np.mean(np.sum(x * np.log(x_hat) + (1.0 - x) * np.log(1.0 - x_hat), axis=1))
        return bce
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

def kl_divergence(mu, log_var):
    """
    KL divergence between N(mu, sigma^2) and N(0, I).
    
    Comment: derive this formula from the definition of KL divergence:
    KL(N(mu, sigma^2) || N(0, I)) = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    This term acts as a regularizer on the latent space, penalizing distributions
    that deviate from the standard normal. It forces the encodings to be close
    to the origin and densely packed.
    
    Args:
        mu (np.ndarray): Mean vectors (batch_size, latent_dim)
        log_var (np.ndarray): Log variance vectors (batch_size, latent_dim)
        
    Returns:
        float: the scalar KL divergence over the batch
    """
    # Sum over latent dimensions, mean over batch
    kl = -0.5 * np.mean(np.sum(1 + log_var - np.square(mu) - np.exp(log_var), axis=1))
    return kl

def elbo_loss(x, x_hat, mu, log_var, beta=1.0, loss_type="bce"):
    """
    Total VAE loss = reconstruction_loss + beta * kl_divergence
    
    Comment: the role of beta in balancing reconstruction vs regularization.
    beta=1.0 is standard VAE; beta>1 is beta-VAE (stronger disentanglement).
    A higher beta forces stronger regularization at the cost of reconstruction fidelity.
    
    Args:
        x (np.ndarray): Original patterns
        x_hat (np.ndarray): Reconstructed patterns
        mu (np.ndarray): Latent means
        log_var (np.ndarray): Latent log variances
        beta (float): KL divergence weight
        loss_type (str): "bce" or "mse"
        
    Returns:
        total_loss (float), reconstruction_term (float), kl_term (float)
    """
    recon_term = reconstruction_loss(x, x_hat, loss_type)
    kl_term = kl_divergence(mu, log_var)
    total_loss = recon_term + beta * kl_term
    return total_loss, recon_term, kl_term

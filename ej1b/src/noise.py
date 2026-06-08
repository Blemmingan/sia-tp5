"""
noise.py

This module contains noise injection functions to corrupt the input patterns
for training the Denoising Autoencoder (DAE). Each function implements a different
noise model to test the autoencoder's robustness.
"""
import numpy as np

def salt_and_pepper(pattern, p):
    """
    Flip each bit independently with probability p. Returns noisy pattern.
    Formula: x_noisy = 1 - x_clean with probability p, otherwise x_noisy = x_clean.
    
    Args:
        pattern (np.ndarray): Original pattern(s) of shape (35,) or (N, 35)
        p (float): Probability of flipping a bit (0 <= p <= 1)
        
    Returns:
        np.ndarray: Noisy pattern(s) with the same shape.
    """
    # Create a mask of the same shape where True means flip the bit
    flip_mask = np.random.rand(*pattern.shape) < p
    
    # Where flip_mask is True, flip the bit (1 -> 0, 0 -> 1) by doing 1 - pattern
    # Where flip_mask is False, keep the original pattern
    noisy_pattern = np.where(flip_mask, 1 - pattern, pattern)
    
    return noisy_pattern

def add_gaussian_noise(pattern, p):
    """
    Add Gaussian noise scaled by p, then clip to [0,1]. Returns noisy float pattern.
    Formula: x_noisy = clip(x_clean + N(0, p^2), 0, 1)
    
    Args:
        pattern (np.ndarray): Original pattern(s) of shape (35,) or (N, 35)
        p (float): Standard deviation of the Gaussian noise
        
    Returns:
        np.ndarray: Noisy pattern(s) with the same shape, values clipped to [0, 1].
    """
    # Generate Gaussian noise with mean 0 and standard deviation p
    noise = np.random.normal(loc=0.0, scale=p, size=pattern.shape)
    
    # Add noise to the pattern and clip values to be between 0 and 1
    noisy_pattern = np.clip(pattern + noise, 0.0, 1.0)
    
    return noisy_pattern

def random_pixel_dropout(pattern, p):
    """
    Set each pixel to 0 with probability p. Returns noisy pattern.
    Formula: x_noisy = 0 with probability p, otherwise x_noisy = x_clean.
    
    Args:
        pattern (np.ndarray): Original pattern(s) of shape (35,) or (N, 35)
        p (float): Probability of setting a pixel to 0 (0 <= p <= 1)
        
    Returns:
        np.ndarray: Noisy pattern(s) with the same shape.
    """
    # Create a mask where True means keep the pixel (probability 1-p)
    # and False means drop the pixel (probability p)
    keep_mask = np.random.rand(*pattern.shape) >= p
    
    # Multiply pattern by the boolean mask (False becomes 0, True becomes 1)
    # This effectively zeroes out pixels where keep_mask is False
    noisy_pattern = pattern * keep_mask
    
    return noisy_pattern

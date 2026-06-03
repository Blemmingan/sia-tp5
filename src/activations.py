"""
activations.py

Implements activation functions and their derivatives.
Each activation function must expose a .forward(x) and .backward(x) method.
"""
import numpy as np

class Sigmoid:
    @staticmethod
    def forward(x):
        '''
        Sigmoid forward pass.
        Args:
            x (np.ndarray): Input values
        Returns:
            np.ndarray: Evaluated sigmoid
        '''
        # Clip to avoid overflow in exp
        x_clipped = np.clip(x, -500, 500)
        return 1.0 / (1.0 + np.exp(-x_clipped))
        
    @staticmethod
    def backward(x):
        '''
        Sigmoid derivative.
        Args:
            x (np.ndarray): Input values
        Returns:
            np.ndarray: Evaluated derivative
        '''
        sig = Sigmoid.forward(x)
        # d/dx sigmoid(x) = sigmoid(x) * (1 - sigmoid(x))
        return sig * (1.0 - sig)

class Tanh:
    @staticmethod
    def forward(x):
        '''
        Tanh forward pass.
        '''
        return np.tanh(x)
        
    @staticmethod
    def backward(x):
        '''
        Tanh derivative.
        '''
        # d/dx tanh(x) = 1 - tanh(x)^2
        return 1.0 - np.tanh(x)**2

class ReLU:
    @staticmethod
    def forward(x):
        '''
        ReLU forward pass.
        '''
        return np.maximum(0, x)
        
    @staticmethod
    def backward(x):
        '''
        ReLU derivative.
        '''
        # d/dx relu(x) = 1 if x > 0 else 0
        return (x > 0).astype(np.float32)

class LeakyReLU:
    alpha = 0.01
    
    @classmethod
    def forward(cls, x):
        '''
        Leaky ReLU forward pass.
        '''
        return np.where(x > 0, x, x * cls.alpha)
        
    @classmethod
    def backward(cls, x):
        '''
        Leaky ReLU derivative.
        '''
        # d/dx leaky_relu(x) = 1 if x > 0 else alpha
        return np.where(x > 0, 1.0, cls.alpha)

class Linear:
    @staticmethod
    def forward(x):
        '''
        Linear forward pass.
        '''
        return x
        
    @staticmethod
    def backward(x):
        '''
        Linear derivative.
        '''
        # d/dx x = 1
        return np.ones_like(x)

def get_activation(name):
    '''
    Helper to get activation class by name.
    '''
    name = name.lower()
    if name == 'sigmoid': return Sigmoid
    elif name == 'tanh': return Tanh
    elif name == 'relu': return ReLU
    elif name == 'leaky_relu': return LeakyReLU
    elif name == 'linear': return Linear
    else: raise ValueError(f"Unknown activation: {name}")

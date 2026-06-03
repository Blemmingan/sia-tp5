"""
optimizers.py

Implements optimizers for training the autoencoder (SGD, Momentum, Adam).
Each optimizer is a class with an update(params, grads) method.
"""
import numpy as np

class SGD:
    def __init__(self, learning_rate=0.01):
        self.lr = learning_rate
        
    def update(self, params, grads):
        '''
        Updates parameters using Standard Gradient Descent.
        
        Args:
            params (list): List of parameter numpy arrays (e.g. [W1, b1, W2, b2, ...])
            grads (list): List of gradient numpy arrays corresponding to params
        '''
        for i in range(len(params)):
            # W = W - lr * dW
            params[i] -= self.lr * grads[i]

class Momentum:
    def __init__(self, learning_rate=0.01, momentum=0.9):
        self.lr = learning_rate
        self.momentum = momentum
        self.velocities = None
        
    def update(self, params, grads):
        '''
        Updates parameters using SGD with Momentum.
        '''
        if self.velocities is None:
            self.velocities = [np.zeros_like(p) for p in params]
            
        for i in range(len(params)):
            # v = momentum * v - lr * dW
            self.velocities[i] = self.momentum * self.velocities[i] - self.lr * grads[i]
            # W = W + v
            params[i] += self.velocities[i]

class Adam:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.t = 0
        
    def update(self, params, grads):
        '''
        Updates parameters using Adam with bias correction.
        '''
        if self.m is None:
            self.m = [np.zeros_like(p) for p in params]
            self.v = [np.zeros_like(p) for p in params]
            
        self.t += 1
        
        for i in range(len(params)):
            # Update biased first moment estimate
            # m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * grads[i]
            
            # Update biased second raw moment estimate
            # v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * (grads[i] ** 2)
            
            # Compute bias-corrected first moment estimate
            m_hat = self.m[i] / (1.0 - self.beta1 ** self.t)
            
            # Compute bias-corrected second raw moment estimate
            v_hat = self.v[i] / (1.0 - self.beta2 ** self.t)
            
            # Update parameters
            # W = W - lr * m_hat / (sqrt(v_hat) + epsilon)
            params[i] -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)

def get_optimizer(name, lr, **kwargs):
    '''
    Helper to get optimizer by name.
    '''
    name = name.lower()
    if name == 'sgd': return SGD(learning_rate=lr)
    elif name == 'momentum': return Momentum(learning_rate=lr, momentum=kwargs.get('momentum', 0.9))
    elif name == 'adam': return Adam(learning_rate=lr, beta1=kwargs.get('beta1', 0.9), beta2=kwargs.get('beta2', 0.999))
    else: raise ValueError(f"Unknown optimizer: {name}")

# Autoencoder Configurations Test

This directory contains `test_configs.sh`, a script designed to train and evaluate multiple configurations of our autoencoder implementation automatically. When executed, it generates comparative results (training logs, model weights, latent space plots, and reconstructions) for each configuration and stores them in the `results_comparative/` directory at the root of the project.

## Tested Configurations

The script evaluates the following four configurations:

### 1. `default`
- **Learning Rate:** `0.01`
- **Optimizer:** `adam`
- **Hidden Activation:** `tanh`
- **Encoder Layers:** `35 -> 16 -> 8 -> 2`
- **Decoder Layers:** `2 -> 8 -> 16 -> 35`
- **Description:** This serves as the baseline configuration. It uses the Adam optimizer which is robust and fast to converge, and hyperbolic tangent (tanh) activation functions which are standard for this type of network to output values between -1 and 1.

### 2. `momentum_opt`
- **Learning Rate:** `0.01`
- **Optimizer:** `momentum`
- **Hidden Activation:** `tanh`
- **Encoder Layers:** `35 -> 16 -> 8 -> 2`
- **Decoder Layers:** `2 -> 8 -> 16 -> 35`
- **Description:** This configuration is identical to the default but replaces the Adam optimizer with a standard Momentum optimizer. It is useful for observing the difference in convergence speed and stability when using a simpler optimization algorithm compared to Adam's adaptive learning rates.

### 3. `relu_activation`
- **Learning Rate:** `0.01`
- **Optimizer:** `adam`
- **Hidden Activation:** `relu`
- **Encoder Layers:** `35 -> 16 -> 8 -> 2`
- **Decoder Layers:** `2 -> 8 -> 16 -> 35`
- **Description:** This configuration substitutes the `tanh` hidden activation function with `relu` (Rectified Linear Unit). ReLU is known to mitigate the vanishing gradient problem, allowing us to evaluate if it yields better reconstruction performance or faster convergence on this specific binary dataset.

### 4. `deep_network`
- **Learning Rate:** `0.01`
- **Optimizer:** `adam`
- **Hidden Activation:** `tanh`
- **Encoder Layers:** `35 -> 16 -> 10 -> 5 -> 2`
- **Decoder Layers:** `2 -> 5 -> 10 -> 16 -> 35`
- **Description:** This configuration features a significantly deeper architecture, introducing more hidden layers to the encoder and decoder. It tests whether a higher capacity network can capture more intricate patterns in the dataset or if it simply leads to overfitting or longer training times without substantial benefits.

### 5. `high_lr`
- **Learning Rate:** `0.1`
- **Optimizer:** `adam`
- **Hidden Activation:** `tanh`
- **Encoder Layers:** `35 -> 16 -> 8 -> 2`
- **Decoder Layers:** `2 -> 8 -> 16 -> 35`
- **Description:** Tests the effect of using a significantly higher learning rate. This might lead to faster initial convergence but risks overshooting the global minimum or failing to converge entirely.

### 6. `low_lr`
- **Learning Rate:** `0.001`
- **Optimizer:** `adam`
- **Hidden Activation:** `tanh`
- **Encoder Layers:** `35 -> 16 -> 8 -> 2`
- **Decoder Layers:** `2 -> 8 -> 16 -> 35`
- **Description:** Tests a more conservative learning rate. We expect much slower convergence, possibly requiring more epochs to reach a low error rate, but with potentially a more stable and finely-tuned final model.

### 7. `shallow_network`
- **Learning Rate:** `0.01`
- **Optimizer:** `adam`
- **Hidden Activation:** `tanh`
- **Encoder Layers:** `35 -> 8 -> 2`
- **Decoder Layers:** `2 -> 8 -> 35`
- **Description:** Reduces the depth of the network by removing the 16-neuron hidden layers. Tests whether a simpler architecture is sufficient to capture the font patterns, reducing training time and the risk of overfitting.

### 8. `asymmetric`
- **Learning Rate:** `0.01`
- **Optimizer:** `adam`
- **Hidden Activation:** `tanh`
- **Encoder Layers:** `35 -> 16 -> 8 -> 2`
- **Decoder Layers:** `2 -> 16 -> 35`
- **Description:** Features an asymmetric architecture where the encoder is deeper than the decoder. This evaluates if the model can still effectively decode from the 2D latent space with a slightly less complex expansion phase.

## Final Output

At the end of the script execution, a Python script (`compare_latent.py`) is run. This generates a final comparative plot named `latent_space_comparison.png` in the `results_comparative/` directory, which visualizes the latent spaces created by all configurations side-by-side.

## How to Run

1. Navigate to this directory (`src/ej1/`).
2. Make sure the script is executable: `chmod +x test_configs.sh`.
3. Run the script: `./test_configs.sh`.
4. Inspect the generated comparisons inside the `results_comparative/` directory at the project root.

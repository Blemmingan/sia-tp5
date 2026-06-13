# Variational Autoencoder (VAE)

This directory contains a complete implementation of a Variational Autoencoder (VAE) from scratch using NumPy.

## What is a VAE?

A standard autoencoder compresses input into a single deterministic point $z$ in latent space. A **Variational Autoencoder (VAE)** instead compresses the input into a probability distribution.

- **Reparameterization Trick:** The encoder outputs a mean ($\mu$) and log variance ($\log \sigma^2$). We sample $z$ as $z = \mu + \sigma \cdot \epsilon$ where $\epsilon \sim \mathcal{N}(0, I)$. This trick separates the randomness ($\epsilon$) from the network parameters, making the entire process differentiable and enabling backpropagation.
- **ELBO Loss:** The Evidence Lower BOund (ELBO) is maximized, meaning we minimize:
  `Loss = Reconstruction Loss + KL Divergence`
- **KL Divergence:** This term regularizes the latent space. It forces the distributions $q(z|x)$ to be close to a standard normal distribution $\mathcal{N}(0, I)$, ensuring that the latent space is continuous and densely packed, which is essential for generation.

## Dataset

The dataset consists of **16 handcrafted 7x7 binary patterns** representing various emojis and symbols. We built this custom dataset to have a visually coherent set of shapes that a VAE can learn to interpolate between.

**Categories:**
1. **Faces:** smiley, sad face, surprised face, neutral face
2. **Shapes:** heart, star, diamond, circle
3. **Arrows:** up, down, left, right
4. **Symbols:** plus, cross (X), checkmark, moon crescent

```
Pattern 1: smiley
| █ █ █ █ █  |
|█         █|
|█ █   █   █|
|█         █|
|██       ██|
|█ █ █ █   █|
| █ █ █ █ █  |

Pattern 5: heart
| ██ ██  |
|███████|
|███████|
| █████  |
|  ███   |
|   █    |
|       |
```

## Architecture

x (49) → Encoder → μ (2), log_var (2)
                     ↓  reparameterize
                    z (2)
                     ↓
         Decoder → x_hat (49)

- **Encoder:** Dense(49, 32) -> Tanh -> Dense(32, 16) -> Tanh -> Dense(16, 2) [for mu], Dense(16, 2) [for log_var]
- **Decoder:** Dense(2, 16) -> Tanh -> Dense(16, 32) -> Tanh -> Dense(32, 49) -> Sigmoid

## How to Run

Use the provided bash script to run different tasks. Results are saved in `outputs/`.

```bash
# Train the VAE, generate all visualizations, and test reconstruction
./ej2/run_tests.sh all

# Just train the model
./ej2/run_tests.sh train

# Generate plots (Latent space, Reconstructions, etc.)
./ej2/run_tests.sh visualize

# Generate 5 new patterns and display them in the terminal as ASCII art
./ej2/run_tests.sh generate 5

# Print the 16 original dataset patterns in the terminal
./ej2/run_tests.sh preview

# Sweep over different Beta values for the KL divergence term
./ej2/run_tests.sh beta-sweep

# Sweep over different Latent Dimensions (2, 4, 8)
./ej2/run_tests.sh latent-sweep

# Clean all generated files
./ej2/run_tests.sh clean
```

## Generation

Because the VAE enforces a regularized latent space, we can generate completely new patterns that "belong to the dataset". We do this by:
1. Sampling $z \sim \mathcal{N}(0, I)$
2. Passing $z$ through the Decoder

The resulting pattern will visually blend properties of the training data.
The **latent traversal plot** demonstrates how smoothly the VAE interpolates between features (e.g., from an arrow pointing left to a star) as we vary a single latent dimension from -3.0 to +3.0.

## Results

Run `./ej2/run_tests.sh all` to populate the `outputs/` directory with visualizations and training logs!

# Exercise 1b — Denoising Autoencoder (DAE)

This directory contains the implementation of a Denoising Autoencoder (DAE) trained to reconstruct clean 5x7 character patterns from noisy inputs.

## Architecture Choice

Unlike the autoencoder in `ej1a`, which used a strictly 2-dimensional latent bottleneck for 2D visualization purposes, the DAE uses a wider latent representation (e.g., 8D). 

**Why 8D?** 
The 2D bottleneck in `ej1a` is a design constraint meant for interpretability, not an optimum for reconstruction. For denoising, the network requires sufficient latent capacity to robustly store the structural representation of characters under severe noise conditions. If the bottleneck is too small, the network loses both noise and structural signal; if it is wider, it has the capacity to filter out noise while preserving the clean representation. We employ a symmetric encoder-decoder structure, which is standard for autoencoders, mapping 35 -> 32 -> 16 -> 8 and back.

## How the Denoising Works

The training procedure works as follows:
1. Every epoch, we take the clean character dataset.
2. We inject fresh noise into the data according to the configured noise function and level (e.g., 10% Salt & Pepper).
3. The model receives the **noisy input** and generates a reconstruction.
4. Crucially, the **loss is calculated against the original, clean input**.

This forces the network to learn to ignore the random perturbations and recover the underlying structure, essentially learning a manifold of "clean" patterns.

## Noise Functions

Three noise models are implemented in `src/noise.py`:

- **Salt & Pepper (`salt_and_pepper`)**: Flips each bit independently with probability `p`. Simulates discrete errors. Formula: `x' = 1 - x` with prob `p`, else `x`.
- **Gaussian (`add_gaussian_noise`)**: Adds continuous Gaussian noise scaled by `p` and clips to `[0,1]`. Formula: `clip(x + N(0, p^2), 0, 1)`.
- **Random Pixel Dropout (`random_pixel_dropout`)**: Sets each pixel to 0 with probability `p`. Simulates missing data. Formula: `x' = 0` with prob `p`, else `x`.

## How to Use `run_tests.sh`

The test runner script handles everything including venv creation, training, evaluation, and visualizations. Run it from the repository root:

```bash
# Train the DAE with default config
./ej1b/run_tests.sh train

# Test reconstruction at all noise levels
./ej1b/run_tests.sh test

# Test at a specific noise level (e.g., 30%)
./ej1b/run_tests.sh test-noise 0.30

# Generate all plots (after training)
./ej1b/run_tests.sh visualize

# Train and evaluate across all noise levels automatically
./ej1b/run_tests.sh noise-sweep

# Train 3 models (one per noise function) and compare
./ej1b/run_tests.sh compare-noise-fns

# Remove outputs
./ej1b/run_tests.sh clean

# Run train + visualize + test sequentially
./ej1b/run_tests.sh all
```

## Results
Run `./ej1b/run_tests.sh all` to populate `outputs/` with plots and the `noise_study.csv` summary.

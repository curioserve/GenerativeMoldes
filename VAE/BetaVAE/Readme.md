# Beta-VAE for Fashion MNIST

This project implements a **Beta-VAE** (β-VAE) model, an extension of the Variational Autoencoder (VAE), designed to learn disentangled latent features in data. The β parameter controls the balance between reconstruction accuracy and latent space disentanglement.

## Overview

- **Goal**: Train a Beta-VAE model on the Fashion MNIST dataset to learn disentangled latent features.
- **Techniques**: Beta-VAE for better separation in the latent space.
- **Dataset**: Fashion MNIST, 28x28 grayscale images of clothing items.

## Model Overview

- **Encoder**: Maps images to a latent space (mean and variance).
- **Latent Space**: Disentangled features representing different image attributes.
- **Decoder**: Reconstructs images from the latent variables.
- **Loss Function**: Combines reconstruction loss and KL divergence, scaled by β.
- **Hyperparameter β**: Controls disentanglement. A higher β increases separation but may reduce reconstruction quality.

## Steps

1. **Data Loading**: Use Fashion MNIST dataset (available via PyTorch’s `torchvision`).
2. **Beta-VAE Architecture**: Implement encoder and decoder networks with the Beta-VAE loss function.
3. **Training**: Train the model while adjusting β for optimal disentanglement.
4. **Evaluation**: Visualize the latent space and evaluate reconstruction quality.

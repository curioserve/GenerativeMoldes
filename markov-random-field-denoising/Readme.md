# Markov Random Field (MRF) for Image Recovery

This project aims to recover the original binary image from a noisy version using a Markov Random Field (MRF). Each pixel in the noisy image has a 0.1 probability of flipping. The task is to minimize the following energy function to estimate the real pixel values:

$$E=\alpha \sum_i  x_i  -\beta \sum_{i,j}x_{i}x_{j} -\gamma \sum_i x_i y_i$$

## Steps

### 1. Initialization

Start with the noisy pixel values as the initial guess for the real image. The noisy image is represented by a binary matrix with values +1 or -1 for each pixel. These noisy values are the initial values for the real image.

### 2. Energy Minimization

To recover the original image, sequentially flip pixel values (+1 or -1) to minimize the energy function. This process continues until convergence, meaning no further flip reduces the energy.

The energy function is composed of three terms:

- **Unary term**: $\alpha \sum_i x_i$, where $x_i$ is the pixel value at index $i$.
- **Pairwise term**: $-\beta \sum_{i,j}x_{i}x_{j}$, which accounts for the interaction between neighboring pixels.
- **Data term**: $-\gamma \sum_i x_i y_i$, where $y_i$ is the noisy pixel value at index $i$.

The goal is to minimize this energy function iteratively.

### 3. Output

Once the energy function is minimized and the pixel values stabilize, the recovered image is displayed. The accuracy of the recovered image can be computed by comparing the estimated pixel values to the ground truth original binary image.


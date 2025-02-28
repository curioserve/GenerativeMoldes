# PixelCNN: A Generative Model

PixelCNN is an autoregressive generative model designed for modeling images by capturing the dependencies between pixels. This model generates new images pixel by pixel, conditioned on previous pixels, which allows it to generate high-quality, coherent images.

## Overview

PixelCNN is a type of autoregressive model that predicts each pixel based on the previous ones. It decomposes the joint probability distribution of pixels into a product of conditional distributions, which is represented as:

$$ P(x) = \prod_i P(x_i | x_{1:i-1}) $$

Where each pixel $x_i$ is conditioned on the previous pixels $x_{1:i-1}$.

Unlike traditional generative models, PixelCNN's architecture is designed to capture complex dependencies between neighboring pixels to generate images in a coherent way.

## PixelCNN Architecture

PixelCNN uses a stack of convolutional layers, each of which is masked to ensure that a pixel at position $(i, j)$ only depends on previously observed pixels. This preserves the autoregressive nature of the model. It models the conditional probability of a pixel given its neighbors and is trained to predict the next pixel conditioned on prior pixels.

![PixelCNN Architecture](https://camo.githubusercontent.com/2b432c6d87633c75685c3703167c0a6b5a6d6592a7ca95540bf02f6de890052c/68747470733a2f2f6c696c69616e77656e672e6769746875622e696f2f6c696c2d6c6f672f6173736574732f696d616765732f706978656c2d636e6e2e706e67)

## Key Concepts

### Autoregressive Models

Autoregressive models predict the next element in a sequence based on prior elements. In PixelCNN, this is applied to image generation, where the model predicts each pixel in the image based on previously generated pixels.

### Masked Convolutions

PixelCNN uses two types of convolution masks to enforce the autoregressive property:

- **Mask A**: Ensures that the current pixel does not depend on future pixels.
- **Mask B**: Ensures that the current pixel only depends on pixels that have already been predicted.

These masked convolutions are key to the model's ability to generate images in an autoregressive fashion.



# MDMLP: Image Classification from Scratch on Small Datasets with MLP

This repository contains an implementation of the MDMLP (Multi-Dimensional MLP) architecture described in the paper ["MDMLP: Image Classification from Scratch on Small Datasets with MLP"](https://arxiv.org/abs/2205.14477) by Lv et al.

## Overview

MDMLP is a lightweight MLP-based architecture designed for training from scratch on small datasets like CIFAR-10. It uses:
- Overlapping patch embeddings
- Multi-dimensional processing (maintaining height, width, channel, and token dimensions)
- An attention visualization tool (MDAttnTool)

## Implementation

This implementation is built with PyTorch and includes:
- Complete MDMLP model architecture
- Training and evaluation code
- Attention visualization

## Results

The implementation achieves:
- 88.06% accuracy on CIFAR-10
- 0.57M parameters
- Better performance than other MLP-based models when trained from scratch

## Usage

To train the model:

python test.py

## Requirements

- PyTorch
- torchvision
- matplotlib
- numpy
- scikit-learn
- einops

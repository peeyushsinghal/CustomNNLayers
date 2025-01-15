# Custom CNN and Transformer Models

This repository contains implementations of a custom Convolutional Neural Network (CNN) for the MNIST dataset and a custom Transformer model for text data. Both models utilize custom layers and are built using PyTorch.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [CNN Training Logs](#cnn-training-logs)
- [Transformer Training Logs](#transformer-training-logs)

## Installation

To install the required packages, you can use pip:

```bash
pip install torch torchvision
```

## Usage

To train the CNN model on the MNIST dataset, run the following command:

```bash
python custom_cnn.py
```

To train the Transformer model on a simple text dataset, run the following command:

```bash
python custom_transformer.py
```


## CNN Training Logs
Train Epoch: 0 [0/60000 (0%)]	Loss: 0.063750
Train Epoch: 0 [6400/60000 (11%)]	Loss: 0.095912
Train Epoch: 0 [12800/60000 (21%)]	Loss: 0.211452
Train Epoch: 0 [19200/60000 (32%)]	Loss: 0.176659
Train Epoch: 0 [25600/60000 (43%)]	Loss: 0.123121
Train Epoch: 0 [32000/60000 (53%)]	Loss: 0.300204
Train Epoch: 0 [38400/60000 (64%)]	Loss: 0.312506
Train Epoch: 0 [44800/60000 (75%)]	Loss: 0.135378
Train Epoch: 0 [51200/60000 (85%)]	Loss: 0.061804
Train Epoch: 0 [57600/60000 (96%)]	Loss: 0.139258

Train Epoch: 1 [0/60000 (0%)]	Loss: 0.126188
Train Epoch: 1 [6400/60000 (11%)]	Loss: 0.145329
Train Epoch: 1 [12800/60000 (21%)]	Loss: 0.160962
Train Epoch: 1 [19200/60000 (32%)]	Loss: 0.060585
Train Epoch: 1 [25600/60000 (43%)]	Loss: 0.172602
Train Epoch: 1 [32000/60000 (53%)]	Loss: 0.225173
Train Epoch: 1 [38400/60000 (64%)]	Loss: 0.188731
Train Epoch: 1 [44800/60000 (75%)]	Loss: 0.163453
Train Epoch: 1 [51200/60000 (85%)]	Loss: 0.264085
Train Epoch: 1 [57600/60000 (96%)]	Loss: 0.057970

Train Epoch: 2 [0/60000 (0%)]	Loss: 0.170645
Train Epoch: 2 [6400/60000 (11%)]	Loss: 0.076135
Train Epoch: 2 [12800/60000 (21%)]	Loss: 0.046482
Train Epoch: 2 [19200/60000 (32%)]	Loss: 0.084306
Train Epoch: 2 [25600/60000 (43%)]	Loss: 0.176785
Train Epoch: 2 [32000/60000 (53%)]	Loss: 0.176272
Train Epoch: 2 [38400/60000 (64%)]	Loss: 0.068932
Train Epoch: 2 [44800/60000 (75%)]	Loss: 0.263448
Train Epoch: 2 [51200/60000 (85%)]	Loss: 0.137503
Train Epoch: 2 [57600/60000 (96%)]	Loss: 0.195460

Train Epoch: 3 [0/60000 (0%)]	Loss: 0.066153
Train Epoch: 3 [6400/60000 (11%)]	Loss: 0.094050
Train Epoch: 3 [12800/60000 (21%)]	Loss: 0.043584
Train Epoch: 3 [19200/60000 (32%)]	Loss: 0.026709
Train Epoch: 3 [25600/60000 (43%)]	Loss: 0.093872
Train Epoch: 3 [32000/60000 (53%)]	Loss: 0.097646
Train Epoch: 3 [38400/60000 (64%)]	Loss: 0.063766
Train Epoch: 3 [44800/60000 (75%)]	Loss: 0.106863
Train Epoch: 3 [51200/60000 (85%)]	Loss: 0.164193
Train Epoch: 3 [57600/60000 (96%)]	Loss: 0.079997

Train Epoch: 4 [0/60000 (0%)]	Loss: 0.092704
Train Epoch: 4 [6400/60000 (11%)]	Loss: 0.026844
Train Epoch: 4 [12800/60000 (21%)]	Loss: 0.160942
Train Epoch: 4 [19200/60000 (32%)]	Loss: 0.100325
Train Epoch: 4 [25600/60000 (43%)]	Loss: 0.186581
Train Epoch: 4 [32000/60000 (53%)]	Loss: 0.087199
Train Epoch: 4 [38400/60000 (64%)]	Loss: 0.111526
Train Epoch: 4 [44800/60000 (75%)]	Loss: 0.062025
Train Epoch: 4 [51200/60000 (85%)]	Loss: 0.211661
Train Epoch: 4 [57600/60000 (96%)]	Loss: 0.120458


## Transformer Training Logs

Batch 0, Loss: 31741.2227
Batch 10, Loss: 29152.0977

Epoch completed. Average loss: 29483.1964

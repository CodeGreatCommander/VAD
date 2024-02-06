# Repository Structure and Usage Guide

Welcome to the Voice Activity Detection (VAD) model repository. This project is organized into different folders, each serving a specific purpose in the development, training, and inference of the VAD model.

## Folders Overview

### 1. `pretrained_models`

This folder contains the necessary components for performing inference using pretrained VAD models. It includes the model architecture, weights, and an inference module implemented in C++. Users can leverage these pretrained models to quickly identify voice activity in audio segments without the need for training.

### 2. `pynote`

The `pynote` folder is dedicated to the training of VAD models. While the primary functionality is present, it is essential to note that extensive testing is still ongoing to enhance and optimize the training process. Users interested in further refining the VAD model or experimenting with different training configurations can explore this folder.

### 3. `utils`

The `utils` folder hosts utilities for preprocessing audio files and classes for efficient dataset access. Proper data preprocessing is crucial for training robust models. This folder includes scripts and modules to facilitate the preparation of audio data for training. Users can adapt these utilities to suit their specific datasets and requirements.

## Prerequisites

Before diving into specific folders, ensure you have a GNU compiler of version 12.3.0 or later installed on your system.

## Getting Started

Follow the instructions in each respective folder to set up, train, and perform inference with the VAD models. Below is a brief guide for each folder:

- **`pretrained_models`**: Use the provided C++ inference module to quickly integrate pretrained models into your applications. Refer to the included code snippet in the README for a simple example.

- **`pynote`**: Explore and experiment with the training process using the Python notebooks in this folder. Note that ongoing testing may result in updates to improve model training.

- **`utils`**: Leverage the preprocessing scripts and dataset access classes in this folder to prepare your audio data for training and inference. Customize these utilities based on your dataset characteristics with the sample one given in Kaldi Data

<!-- ## License -->



We appreciate your interest in the VAD model repository. If you encounter any issues or have suggestions for improvements, please don't hesitate to open an issue or reach out to the maintainers. Happy coding!

# Voice Activity Detection (VAD) Model

This repository contains the inference module for a Voice Activity Detection (VAD) model, implemented in C++. The VAD model is designed to identify whether an input audio segment contains human speech (voice activity) or is background noise (non-voice activity). The model architecture incorporates SincNet layers for efficient feature extraction, LSTM layers for capturing temporal dependencies, and linear layers for final classification.

## Model Architecture

The VAD model consists of the following key components:

1. **SincNet Layers (Convolutions):**
   - SincNet layers are utilized for effective feature extraction from the input audio signal.
   - These layers employ parametrized sinc functions to learn filters that are particularly well-suited for audio processing.

2. **LSTM Layers:**
   - Long Short-Term Memory (LSTM) layers are employed to capture temporal dependencies in the audio sequence.
   - LSTMs help the model understand the context and sequential patterns in the input data.

3. **Linear Layers:**
   - Linear layers are used for the final classification task, distinguishing between voice activity and non-voice activity.
   - These layers make the model capable of providing binary predictions for each input segment.

## Prerequisites

Ensure you have a GNU compiler of version 12.3.0 or later installed on your system.

## Installation

1. Clone the repository to your local machine.

    ```bash
    git clone https://github.com/CodeGreatCommander/VAD
    cd VAD/inference/pretrained_models
    ```

2. Follow the instructions in the subsequent sections to set up and use the VAD inference module.

## Usage

The VAD inference module provides a streamlined process for performing voice activity detection on audio segments. The pretrained models are located in the `./inference/pretrained_models` directory to further explore the model please use the README of the desired folder section
## License
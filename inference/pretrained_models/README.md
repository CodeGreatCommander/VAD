# Inference Module for Voice Activity Detection (VAD)

This repository hosts a sophisticated inference module tailored for Voice Activity Detection (VAD). The module includes pretrained models and extends its functionality to support both individual file and folder-based inference scenarios.

## Prerequisites

There are no explicit prerequisites to accessing the model except for the GCC version requirement, but it is suggested if you have a pre installed onnxruntime to create a directory in the root folder of the project by the name of `dependencies` and copy the `onnxruntime` in it otherwise it will be self handled in the further steps

## Dependencies

This project relies on the following crucial libraries:

1. **libtorch**
2. **libsndfile**
3. **onnxruntime**

## Data.txt Format

For seamless inference from a single file or folder, adhere to the following `data.txt` file format:

1. **First Line:** Path to the file or folder for inference.
2. **Second Line:** Path to the model.
3. **Third Line:** Path to the destination for storing final inference results.

## Running the Code

Initiate dependencies installation and code compilation with the following command:

```bash
make initialize  
```
For single-file inference, execute:
```bash
make infer FILE=/home/rohan/VAD/dataset/third_dihard_challenge_eval/data/flac/DH_EVAL_0001.flac MODEL=pyannote
```
For folder-based inference,execute:
```bash
make batch DIR=/home/rohan/VAD/dataset/third_dihard_challenge_eval/data/flac/ MODEL=pyannote
```
Ensure to substitute the placeholders with project-specific information.

## Understanding ONNX
This project seamlessly integrates the Open Neural Network Exchange (ONNX) format, fostering interoperability across diverse AI frameworks. ONNX serves as a standardized representation of models, streamlining their sharing and deployment on various platforms. This choice enhances flexibility and ease of integration into different inference engines.
## Understanding Libsndfile Audio File Handling
The VAD module employs `libsndfile` for seamless audio file handling. This C library supports various formats, simplifying loading and processing tasks during both training and inference in the VAD module.

Feel encouraged to explore the codebase, customize it according to your project's requirements, and contribute to the continual refinement of this VAD inference module. Should you encounter issues or have valuable suggestions, kindly open an issue or reach out to the maintainers.
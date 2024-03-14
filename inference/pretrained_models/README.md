# Inference Module for Voice Activity Detection (VAD)

This repository hosts a sophisticated inference module tailored for Voice Activity Detection (VAD). The module includes pretrained models and extends its functionality to support both individual file and folder-based inference scenarios.

## Prerequisites

There are no explicit prerequisites to accessing the model except for the GCC version requirement, but it is suggested if you have a pre installed onnxruntime to create a directory in the root folder of the project by the name of `dependencies` and copy the `onnxruntime` in it otherwise it will be self handled in the further steps

## Dependencies

This project relies on the following crucial libraries:

1. **libtorch**
2. **libsndfile**
3. **onnxruntime**


## Executing the Code

To set up the project and compile the code, initiate the installation of dependencies with the following command:

```bash
make initialize  
```
For inference on a single file, use the following command. Replace FILE with the path to your audio file and MODEL with the model you wish to use:

```bash
make infer FILE=/home/rohan/VAD/dataset/third_dihard_challenge_eval/data/flac/DH_EVAL_0001.flac MODEL=pyannote
```
For inference on multiple files in a directory, use the following command. Replace DIR with the path to your directory and MODEL with the model you wish to use:

```bash
make batch DIR=/home/rohan/VAD/dataset/third_dihard_challenge_eval/data/flac/ MODEL=pyannote
```
To evaluate the model's performance on a single file, use the following command. Replace FILE with the path to your output file, RTTM with the path to your RTTM file, and AUDIO with the path to your audio file:
```bash
make evaluate FILE=/home/rohan/VAD/inference/pretrained_models/output/output.txt RTTM=/home/rohan/VAD/dataset/data_tesing/an4_diarize_test.rttm AUDIO=/home/rohan/VAD/dataset/data_tesing/an4_diarize_test.wav
```

To evaluate the model's performance on multiple files in a directory, use the following command. Replace DIR with the path to your output directory, RTTM_DIR with the path to your RTTM directory, and AUDIO_DIR with the path to your audio directory:
```bash
make evaluate_batch DIR=/home/rohan/VAD/inference/pretrained_models/output RTTM_DIR=/home/rohan/VAD/dataset/third_dihard_challenge_eval/data/rttm AUDIO_DIR=/home/rohan/VAD/dataset/third_dihard_challenge_eval/data/flac
```

Please ensure to replace all placeholders with the appropriate paths and model names specific to your project.

## Understanding ONNX
This project seamlessly integrates the Open Neural Network Exchange (ONNX) format, fostering interoperability across diverse AI frameworks. ONNX serves as a standardized representation of models, streamlining their sharing and deployment on various platforms. This choice enhances flexibility and ease of integration into different inference engines.
## Understanding Libsndfile Audio File Handling
The VAD module employs `libsndfile` for seamless audio file handling. This C library supports various formats, simplifying loading and processing tasks during both training and inference in the VAD module.

Feel encouraged to explore the codebase, customize it according to your project's requirements, and contribute to the continual refinement of this VAD inference module. Should you encounter issues or have valuable suggestions, kindly open an issue or reach out to the maintainers.
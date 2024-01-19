# Inference Module

This project provides an inference module for voice activity detection (VAD). It includes pretrained models and supports inference from a single file or an entire folder.

## Prerequisites

If you have a pre-installed and built library of `onnxruntime`, please create a directory named `dependencies` in the root project folder and move the `onnxruntime` folder to this path.

## Dependencies

This project depends on the following libraries:

1. `libtorch`
2. `libsndfile`
3. `onnxruntime`

## Data.txt Format

For inference from a single file or folder, the `data.txt` file should be formatted as follows:

1. First line: Path to the file or folder to be inferred from.
2. Second line: Path to the model.
3. Third line: Path to the file or folder where the final inference results will be stored.

## Running the Code

To install dependencies and compile the code, run the following command:

```bash
make initialize  
```
To infer from a single file, use the following command:
```bash
make infer 
```
To infer from a complete folder, use the following command:
```bash
make inferfolder
```

Remember to replace the placeholders with the actual information about your project.
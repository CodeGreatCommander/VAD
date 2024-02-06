from pyannote.audio.models.blocks.sincnet import SincNet
import torch
import onnx

sincnet_model = SincNet()

dummy_input = torch.randn(1, 1, 16000)  # Adjust input shape to match your model's input

# Export the model to ONNX
torch.onnx.export(sincnet_model, dummy_input, "sincnet_model.onnx", verbose=True)
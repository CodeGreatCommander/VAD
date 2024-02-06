#include <onnxruntime/core/session/onnxruntime_cxx_api.h>
#include <torch/torch.h>
#include <cstdlib>
namespace sincnet{

    class SincNet : public torch::nn::Module {
    public:
        SincNet() {
            // Run Python script to make the model
            std::system("python3 /home/rohan/VAD/inference/pynote/complete_work/model/sincnet/sincnet.py");

            // Initialize ONNX Runtime and load SincNet model
            Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "SincNetModel");
            Ort::SessionOptions session_options;
            sincnet_model = std::make_unique<Ort::Session>(env, "/home/rohan/VAD/inference/pynote/complete_work/model/sincnet/sincnet.onnx", session_options);
        }

        torch::Tensor forward(torch::Tensor x) {
            // Create an OrtMemoryInfo object
            Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

            // Convert PyTorch tensor to ONNX tensor
            Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, x.data_ptr<float>(), x.numel(), x.sizes().data(), x.dim());

            // Run SincNet model
            std::vector<const char*> input_names = {"input"};
            std::vector<const char*> output_names = {"output"};
            std::vector<Ort::Value> output_tensors = sincnet_model->Run(Ort::RunOptions{nullptr}, input_names.data(), &input_tensor, 1, output_names.data(), 1);

            // Convert ONNX tensor to PyTorch tensor
            Ort::Value& output_tensor = output_tensors.front();
            torch::Tensor output = torch::from_blob(output_tensor.GetTensorMutableData<float>(), {output_tensor.GetTensorTypeAndShapeInfo().GetShape()});

            // Continue with the rest of your PyTorch model
            // ...

            return output;
        }

    private:
        std::unique_ptr<Ort::Session> sincnet_model;
    };
}
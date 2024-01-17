#include <torch/torch.h>
class ParamSincFB : public torch::nn::Module {
public:
    ParamSincFB(int in_channels, int out_channels, int kernel_size, int sample_rate) {
        // Define your sinc filter parameters
        weight = register_parameter("weight", torch::randn({out_channels, in_channels, kernel_size}));
        // Example: Initialize other parameters, e.g., bias if needed
        // bias = register_parameter("bias", torch::zeros({out_channels}));
        // Initialize other parameters based on your requirements
        // ...
        // Set other member variables as needed
        this->in_channels = in_channels;
        this->out_channels = out_channels;
        this->kernel_size = kernel_size;
        this->sample_rate = sample_rate;
    }

    torch::Tensor forward(torch::Tensor x) {
        // Implement the forward pass for the sinc filter
        // Example: Use torch::sinc function to compute sinc function
        torch::Tensor sinc_weights = torch::sinc(weight * (x - (kernel_size - 1) / 2.0) / sample_rate);
        return sinc_weights.sum(2) / (2 * (kernel_size - 1) / sample_rate);
    }

private:
    torch::Tensor weight;
    // Other parameters or member variables as needed
    int in_channels;
    int out_channels;
    int kernel_size;
    int sample_rate;
};


class Encoder : public torch::nn::Module {
public:
    Encoder() : filterbank(nullptr){
        filterbank = std::make_shared<ParamSincFB>(1,1,251,16000);//Parameters
        register_module("filterbank", filterbank);
    }

    torch::Tensor forward(torch::Tensor x) {
        x = filterbank->forward(x);
        // You may need to implement additional operations for the Encoder
        // Example: apply convolution, activation, etc.
        return x;
    }

private:
    std::shared_ptr<ParamSincFB> filterbank;
};

class SincNet : public torch::nn::Module {
public:
    SincNet():wav_norm1d(register_module("wav_norm1d", torch::nn::InstanceNorm1d(torch::nn::InstanceNorm1dOptions(1).eps(1e-5).momentum(0.1).track_running_stats(false)))),conv1d(register_module("conv1d", torch::nn::Sequential(
            Encoder(),
            torch::nn::Conv1d(80, 60, 5),
            torch::nn::Conv1d(60, 60, 5)
        ))),pool1d(register_module("pool1d", torch::nn::Sequential(
            torch::nn::MaxPool1d(torch::nn::MaxPool1dOptions(3))
        ))),norm1d(register_module("norm1d", torch::nn::Sequential(
            torch::nn::InstanceNorm1d(torch::nn::InstanceNorm1dOptions(80).eps(1e-5).momentum(0.1).track_running_stats(false)),
            torch::nn::InstanceNorm1d(torch::nn::InstanceNorm1dOptions(60).eps(1e-5).momentum(0.1).track_running_stats(false)),
            torch::nn::InstanceNorm1d(torch::nn::InstanceNorm1dOptions(60).eps(1e-5).momentum(0.1).track_running_stats(false))
        ))){
    }

    torch::Tensor forward(torch::Tensor x) {
        x = wav_norm1d(x);
        for (int i = 0; i < conv1d->size(); ++i) {
            x = torch::relu((*conv1d)[i]->as<torch::nn::Conv1d>()->forward(x));
            x = (*pool1d)[i]->as<torch::nn::MaxPool1d>()->forward(x);
            x = (*norm1d)[i]->as<torch::nn::InstanceNorm1d>()->forward(x);
        }
        return x;
    }

private:
    torch::nn::InstanceNorm1d wav_norm1d;
    torch::nn::Sequential conv1d, pool1d, norm1d;
};


class PyanNet : public torch::nn::Module {
public:
    PyanNet():sincnet(register_module("sincnet", std::make_shared<SincNet>())),
    lstm(register_module("lstm", torch::nn::LSTM(torch::nn::LSTMOptions(60, 128).num_layers(2).dropout(0.1)))),
    linear(register_module("linear", torch::nn::Sequential(
            torch::nn::Linear(256, 128),
            torch::nn::Linear(128, 128)
        ))),
        classifier(register_module("classifier", torch::nn::Linear(128, 1))),activation(torch::nn::Sigmoid()){
    }

    torch::Tensor forward(torch::Tensor x) {
        x = sincnet->forward(x);
        x = std::get<0>(lstm->forward(x));
        x = torch::relu(linear->forward(x));
        x = classifier->forward(x);
        x = activation->forward(x);
        return x;
    }

private:
    std::shared_ptr<SincNet> sincnet;
    torch::nn::LSTM lstm;
    torch::nn::Sequential linear;
    torch::nn::Linear classifier;
    torch::nn::Sigmoid activation;
};

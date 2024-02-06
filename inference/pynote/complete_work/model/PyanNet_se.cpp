#include <torch/torch.h>
#include <map>
#include <string>



namespace PyanNet_SE{

    struct SINCNET_DEFAULTS{
        int32_t stride;
        SINCNET_DEFAULTS(int32_t stride=10):stride(stride){}
    };

    struct LSTM_DEFAULTS{
        int32_t hidden_size;
        int32_t num_layers;
        bool bidirectional;
        bool monolithic;
        LSTM_DEFAULTS(int32_t hidden_size=128, int32_t num_layers=2, bool bidirectional=true, bool monolithic=true):hidden_size(hidden_size), num_layers(num_layers), bidirectional(bidirectional), monolithic(monolithic){}
    };

    struct LINEAR_DEFAULTS{
        int32_t hidden_size;
        int32_t num_layers;
        LINEAR_DEFAULTS(int32_t hidden_size=128, int32_t num_layers=2):hidden_size(hidden_size), num_layers(num_layers){}
    };

    class PyanNet:public torch::nn::Module{
        public:
        PyanNet(int64_t sample_rate=1600, int64_t num_channels=1) {
            // Define SincNet layer
            sincnet = register_module("sincnet", SincNet(/* Add parameters */));

            // LSTM parameters
            int64_t lstm_input_size = 60; // Change this according to your input size
            int64_t lstm_hidden_size = 128;
            int64_t lstm_num_layers = 2;
            bool lstm_bidirectional = true;
            bool lstm_monolithic = true;

            // Define LSTM layers
            if (lstm_monolithic) {
                lstm = register_module("lstm", torch::nn::LSTM(
                    torch::nn::LSTMOptions(lstm_input_size, lstm_hidden_size)
                        .num_layers(lstm_num_layers)
                        .bidirectional(lstm_bidirectional)
                        .batch_first(true)
                ));
            } else {
                // Implement multi-layer LSTM
                // Note: You'll need to add the dropout layer if lstm["num_layers"] > 1
                // Implement as per your logic in Python code
            }

            // Linear parameters
            int64_t linear_hidden_size = 128;
            int64_t linear_num_layers = 2;

            // Define linear layers
            if (linear_num_layers > 0) {
                linear = register_module("linear", torch::nn::Sequential());
                // Add linear layers as per your pairwise logic
                // Use torch::nn::Linear and torch::nn::Functional::ReLU
            }

            // Define classifier (final linear layer)
            classifier = register_module("classifier", torch::nn::Linear(
                /* input_features */ (linear_num_layers > 0) ? linear_hidden_size : lstm_hidden_size * (lstm_bidirectional ? 2 : 1),
                /* output_features */ 1
            ));

            // Define activation function
            activation = torch::nn::Functional(torch::nn::Sigmoid());
        }

        torch::Tensor forward(torch::Tensor input) {
            // Process input through layers sequentially
            input = sincnet->forward(input);
            input = std::get<0>(lstm->forward(input)); // Forward through LSTM

            // Process through linear layers if applicable
            if (linear) {
                input = linear->forward(input);
            }

            // Apply classifier and activation
            input = classifier->forward(input);
            input = torch::sigmoid(input);
            return input;
        }

        private:
            torch::nn::Sequential sincnet{ nullptr }, linear{ nullptr };
            torch::nn::LSTM lstm{ nullptr };
            torch::nn::Linear classifier{ nullptr };
            torch::nn::Functional activation;

    };
} // namespace PyanNet

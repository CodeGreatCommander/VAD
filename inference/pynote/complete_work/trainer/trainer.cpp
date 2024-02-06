#include <torch/torch.h>
#include <iostream>
#include "/home/rohan/VAD/inference/pynote/complete_work/model/PyanNet.cpp"
namespace trainer{

    class Trainer {
    public:
        Trainer(const std::map<std::string, std::string>& train_config, const std::map<std::string, std::string>& model_config)
            : iteration(0) {

            std::string model_name = train_config.at("model_name");
            std::map<std::string, std::string> opt_param = {
                {"optim_type", "Adam"},
                {"learning_rate", "1e-4"},
                {"max_grad_norm", "10"}
            };  // Default optimizer parameters
            
            // Update with train_config values if provided
            for (const auto& pair : train_config) {
                opt_param[pair.first] = pair.second;
            }

            // Determine the device (CPU or CUDA)
            std::vector<int64_t> gpus = { -1 }; // Change this with your GPU indices
            torch::Device device = torch::kCPU;
            if (gpus[0] != -1 && torch::cuda::is_available()) {
                device = torch::Device(torch::kCUDA, gpus[0]);
            }

            // Instantiate your model using the provided model configuration
            PyanNet::PyanNet model;
            model.to(device);

            // Printing the model
            std::cout << model << std::endl;

            // Set the model to training mode
            model.train();

            // Set the learning rate
            double learning_rate = std::stod(opt_param.at("learning_rate"));

            // Choose optimizer based on the optim_type
            optimizer = new torch::optim::Adam(model.parameters(), torch::optim::AdamOptions(learning_rate).betas({0.9, 0.999}).weight_decay(0.0));
            

            // Set the scheduler (if lr_scheduler key exists in opt_param)
            if (opt_param.find("lr_scheduler") != opt_param.end()) {
                // Initialize the scheduler with specific options
                // Note: You need to extract and convert the lr_scheduler options here
                scheduler = std::make_shared<torch::optim::lr_scheduler::StepLR>(*optimizer, /*step_size=*/30, /*gamma=*/0.1);
            } else {
                scheduler = nullptr;
            }
        }

        // Define the step function to perform a training step
        std::tuple<int, std::map<std::string, float>, float> step(torch::Tensor input, int iteration = -1) {
            // Perform a training step using input tensor
            // Placeholder for training logic

            // Sample outputs to match the Python implementation
            std::map<std::string, float> loss_detail = {{"VAD loss", 0.0}};
            float learning_rate = 0.0;

            // Update iteration count
            if (iteration != -1) {
                iteration = iteration + 1;
            } else {
                iteration += 1;
            }

            return std::make_tuple(iteration, loss_detail, learning_rate);
        }

        // Function to save model checkpoint
        void save_checkpoint(const std::string& checkpoint_path) {
            // Save model, optimizer state, and iteration count to a file
            // Placeholder for checkpoint saving logic
            std::cout << "Saved state dict. to " << checkpoint_path << std::endl;
        }

        // Function to load model checkpoint
        int load_checkpoint(const std::string& checkpoint_path) {
            // Load model, optimizer state, and iteration count from a file
            // Placeholder for checkpoint loading logic
            std::cout << "Loaded pretrained model from " << checkpoint_path << std::endl;
            return 0; // Return iteration count
        }

    private:
        torch::optim::Optimizer* optimizer;
        torch::optim::lr_scheduler::StepLR* scheduler;
        int iteration;
    };

}
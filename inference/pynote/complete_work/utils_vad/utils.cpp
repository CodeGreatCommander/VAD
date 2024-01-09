#include <torch/torch.h>
#include <vector>
#include <numeric>
#include <algorithm>

namespace utils_vad_utils{
        torch::Tensor downsample(const torch::Tensor& input, const std::vector<int64_t>& npts) {
        auto num_channels = input.size(0);
        auto original_lengths = input.size(1);

        std::vector<torch::Tensor> downsampled_channels;
        downsampled_channels.reserve(num_channels);

        for (int64_t channel_idx = 0; channel_idx < num_channels; ++channel_idx) {
            auto channel = input[channel_idx];

            // Generate indices for downsampling
            torch::Tensor indices = torch::arange(original_lengths, torch::kFloat32)
                                        .to(torch::kFloat32)
                                        .mul(original_lengths - 1)
                                        .div(npts[channel_idx] - 1)
                                        .round()
                                        .to(torch::kInt64);

            // Perform downsampling using gather
            torch::Tensor downsampled = torch::gather(channel, 0, indices);

            downsampled_channels.push_back(downsampled);
        }

        // Stack downsampled channels into a single tensor
        torch::Tensor downsampled_output = torch::stack(downsampled_channels);

        return downsampled_output;
    }
}
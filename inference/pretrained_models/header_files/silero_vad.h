#pragma once

#include <vector>
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>

struct silero{
    std::vector<Ort::Value> ort_inputs;
    std::vector<float> input_tensor,_h,_c;
    std::vector<int64_t> sr;
    int64_t input_node_dims[2]={},sr_node_dims[1] = {1},hc_node_dims[3] = {2, 1, 64};
    unsigned int size_hc;
};

void initialise_silero();
#pragma once
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>
#include <iostream>
#include <vector>
#include <string>
#include <torch/torch.h>
#include <fstream>
#include <filesystem>
#include <chrono>
#include <set>
#include "../header_files/utils.h"
#include "../../utils/kaldi_data.h"
#include "../header_files/proximityFunction.h"
#include <omp.h>

using namespace std;

std::pair<double,std::vector<float>> inference(std::vector<float>& audio_data,Ort::Session& model,const int chunk_size,const int sampling_rate,const float threshold,const double duration_seconds,const double initial_duration_seconds=0);
void inference_single(const std::string& input_file,Ort::Session& model,const std::string& output_file);
void inference_batch(const std::string& input_folder,Ort::Session& model,const std::string& output_folder);
void infer(const std::string& input,const std::string& model_type,bool batch);
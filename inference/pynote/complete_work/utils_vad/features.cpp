#include <vector>
#include <string>
#include <fstream>
#include <cmath>
#include <iostream>
namespace features{
    float rms(const std::vector<float>& data){
        float sum = 0;
        for(int i=0; i<data.size(); i++){
            sum += data[i]*data[i];
        }
        return std::sqrt(sum/data.size());
    }
    std::vector<float> computeNonlinearEnergy(const std::vector<std::vector<float>>& signal_data) {
        std::vector<float> teager_energy;
        size_t n_samples = signal_data[0].size(); // Assuming all rows have the same number of samples

        for (const auto& row : signal_data) {
            float energy_sum = 0.0f;
            for (size_t i = 1; i < n_samples - 1; ++i) {
                float teager_value = row[i] * row[i] - row[i - 1] * row[i + 1];
                energy_sum += teager_value;
            }
            teager_energy.push_back(energy_sum / static_cast<float>(n_samples - 2));
        }

        return teager_energy;
    }
    float ncc(const std::vector<float>& data0, const std::vector<float>& data1) {
        if (data0.size() != data1.size()) {
            std::cerr << "Error: Input arrays must have the same size." << std::endl;
            return 0.0f;
        }

        size_t size = data0.size();
        float sum = 0.0f;

        for (size_t i = 0; i < size; ++i) {
            sum += data0[i] * data1[i];
        }
        std::vector<float> data0_norm = norm_data(data0);
        std::vector<float> data1_norm = norm_data(data1);
        float norm_product = 0.0f;
        for (size_t i = 0; i < size; ++i) {
            norm_product += data0_norm[i] * data1_norm[i];
        }

        return (1.0f / (size - 1)) * sum / norm_product;
    }
    std::vector<float> norm_data(const std::vector<float> &x){
        float sumsq = 0,sum = 0;
        for(int i=0; i<x.size(); i++){
            sumsq += x[i]*x[i];
            sum += x[i];
        }
        float mean = sum/x.size(), std = std::sqrt(sumsq/x.size() - mean*mean);
        std::vector<float> y(x.size());
        for(int i=0; i<x.size(); i++){
            y[i] = (x[i] - mean)/std;
        }
        return y;
    }

    

    std::vector<std::string> readRTTM(const std::string rttm_path){
        std::vector<std::string> rttm;
        std::ifstream rttm_file(rttm_path);
        if(!rttm_file.is_open()){
            throw std::runtime_error("Failed to open rttm file");
        }
        std::string line;
        while(std::getline(rttm_file, line)){
            rttm.push_back(line.substr(0, line.length()-1));
        }
        return rttm;
    }
}
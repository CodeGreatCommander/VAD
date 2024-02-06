#include "kaldi_data.h"
namespace kaldi_data{

    std::map<std::string,std::string> loadWavSCP(const std::string wav_scp_path){
        std::map<std::string, std::string> wav_scp;
        std::ifstream wav_scp_file(wav_scp_path);
        if(!wav_scp_file.is_open()){
            throw std::runtime_error("Failed to open wav.scp file");
        }
        std::string line;
        while(std::getline(wav_scp_file, line)){
            std::istringstream iss(line);
            //This part marks trimming the line
            line.erase(line.begin(), std::find_if(line.begin(), line.end(), [](unsigned char ch) {
                return !std::isspace(ch);
            }));
            line.erase(std::find_if(line.rbegin(), line.rend(), [](unsigned char ch) {
                return !std::isspace(ch);
            }).base(), line.end());
            //---------------------------------
            std::string key, value;
            std::getline(iss, key, ' ');
            std::getline(iss, value);
            wav_scp[key] = value;
        }
        return wav_scp;
    }
    std::vector<std::vector<float>> deinterleave(const std::vector<float>& data, int channels) {
        std::vector<std::vector<float>> result(channels, std::vector<float>(data.size() / channels));
        for (size_t i = 0; i < data.size(); ++i) {
            result[i % channels][i / channels] = data[i];
        }
        return result;
    }
    std::pair<std::vector<std::vector<float>>,int> loadWav(const std::string wav_path,float start_time,float end_time){
        /*TODO: handle case when file name ends with | or - */
        SndfileHandle file(wav_path);
        if(file.error()){
            throw std::runtime_error("Failed to open wav file");
        }
        int samplerate=file.samplerate();
        sf_count_t start_frame = start_time * samplerate;
        sf_count_t end_frame = (end_time < 0) ? file.frames() : end_time * samplerate;
        sf_count_t num_frames = end_frame - start_frame;
        std::vector<float> data(num_frames);
        if (num_frames < 0) {
            throw std::runtime_error("Invalid time range");
        }
        file.seek(start_frame, SEEK_SET);
        file.readf(data.data(), num_frames);
        return {deinterleave(data,file.channels()),samplerate};
    }

    std::map<std::string,float> loadReco2Dur(const std::string reco2dur_path){
        std::map<std::string, float> reco2dur;
        std::ifstream reco2dur_file(reco2dur_path);
        if(!reco2dur_file.is_open()){
            throw std::runtime_error("Failed to open reco2dur file");
        }
        std::string line;
        while(std::getline(reco2dur_file, line)){
            std::istringstream iss(line);
            std::string key;
            float value;
            iss>>key>>value;
            reco2dur[key] = value;
        }
        return reco2dur;
    }

    KaldiData::KaldiData(std::string data_dir):data_dir(data_dir){
        wavs=loadWavSCP(data_dir+"/wav.scp");
        reco2dur=loadReco2Dur(data_dir+"/reco2dur");
    }
    
    std::pair<std::vector<std::vector<float>>,int> KaldiData::loadWav(std::string reco,float start_time,float end_time){
        return kaldi_data::loadWav(wavs[reco],start_time,end_time);
    }
}
// #include <iostream>
// int main(){
//     auto x=kaldi_data::loadWav("/home/rohan/VAD/dataset/record_trial.flac");
//     std::cout<<x.second<<std::endl;
//     return 0;
// }
//compile g++ -g -o kaldi kaldi_data.cpp -I/usr/local/include -lsndfile
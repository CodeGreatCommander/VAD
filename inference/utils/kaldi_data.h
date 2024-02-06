#pragma once

#include <string>
#include <map>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <vector>
#include <sndfile.hh>

namespace kaldi_data {
    std::map<std::string,std::string> loadWavSCP(const std::string wav_scp_path);
    std::vector<std::vector<float>> deinterleave(const std::vector<float>& data, int channels) ;
    std::pair<std::vector<std::vector<float>>,int> loadWav(const std::string wav_path,float start_time=0,float end_time=-1);
    std::map<std::string,float> loadReco2Dur(const std::string reco2dur_path);
    class KaldiData{
        public:
        std::string data_dir;
        std::map<std::string,std::string> wavs;
        std::map<std::string,float> reco2dur;
        KaldiData(std::string data_dir);
        
        std::pair<std::vector<std::vector<float>>,int> loadWav(std::string reco,float start_time=0,float end_time=-1);
    };
}
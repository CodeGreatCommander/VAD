#include <torch/torch.h>
#include <string>
#include "../../../utils/kaldi_data.cpp"
#include "../utils_vad/features.cpp"
#include <vector>
#include <tuple>
#include <map>
#include <iostream>
#include <algorithm>
#include <filesystem>


namespace dataLoaderAug{

    std::vector<std::string> GetFilesInDirectoryWithPattern(std::string noise_path,std::string pattern ){
        // Example noise_path = "/path/to/noise_directory";
        // Example pattern = "*.flac";

        std::vector<std::string> noise_files;

        for (const auto& entry : std::filesystem::directory_iterator(noise_path)) {
            if (std::filesystem::is_regular_file(entry) &&
                entry.path().extension() == pattern.substr(1)) {
                noise_files.push_back(entry.path().string());
            }
        }
        return noise_files;
    }


    struct Segment{
        float start_time,end_time;
        Segment(float start_time,float end_time):start_time(start_time),end_time(end_time){}
        bool Intersect(const Segment& other)const{
            return (start_time<=other.end_time && other.start_time<=end_time);
        }
        Segment operator&(const Segment& other)const{
            return Segment(std::max(start_time,other.start_time),std::min(end_time,other.end_time));
        }
    };


    class Dataset:public torch::data::datasets::Dataset<Dataset>{
        public:


        std::string data_path,rttm_path,noise_path;
        float chunk_size,chunk_shift;
        int rate;
        size_t total_chunk;
        std::vector<std::tuple<std::string,size_t,size_t,size_t>> chunk_indices;
        kaldi_data::KaldiData kaldi_obj;
        std::vector<float> snr;
        std::vector<std::string> noise_files;
        Dataset(std::string data_path,std::string noise_path,std::string rttm_path,float chunk_size=5,float chunk_shift=4,int rate=16000,std::vector<float> snr={-5,0,5}):
        snr(snr),noise_path(noise_path),data_path(data_path),rttm_path(rttm_path),chunk_size(chunk_size),chunk_shift(chunk_shift),rate(rate),kaldi_obj(kaldi_data::KaldiData(data_path)),noise_files(GetFilesInDirectoryWithPattern(noise_path,"*.flac")){
            total_chunk=0;


            for(const std::pair<std::string,std::string>& rec:kaldi_obj.wavs){
                size_t num_chunks=(size_t)((kaldi_obj.reco2dur[rec.first]-chunk_size)/chunk_shift+1);
                auto data=kaldi_data::loadWav(rec.second,0,rate);
                size_t num_channels=data.first.size();

                if(num_channels>100)num_channels=1;
                for(size_t ch=0;ch<num_channels;ch++){

                    for(size_t cu=0;cu<num_chunks;cu++){

                        if(cu*chunk_shift+chunk_size<=kaldi_obj.reco2dur[rec.first]){
                            chunk_indices.push_back(std::make_tuple(rec.first,ch,cu*chunk_shift,cu*chunk_shift+chunk_size));
                        }
                        if(ch==0)
                        total_chunk++;
                    }
                }
            }
            std::cout<<"Total chunks: "<<total_chunk<<std::endl;
        }


        std::vector<int> getMaskFromRTTM(std::string rec_id,size_t num_sample,float time_start,float time_end){
            std::string rttm_path=rttm_path+rec_id+".rttm";
            std::vector<std::string> rttm=features::readRTTM(rttm_path);
            std::vector<int> mask(num_sample,0);
            Segment seg(time_start,time_end);

            for(const std::string& line:rttm){

                std::istringstream iss(line);
                std::string speaker,type,rec,start_time,duration,channel,spkr,label;
                iss>>speaker>>type>>rec>>start_time>>duration>>channel>>spkr>>label;
                Segment curr(std::stof(start_time),std::stof(start_time)+std::stof(duration));
                if(seg.Intersect(curr)){

                    Segment  intersect=seg&curr;
                    size_t start_index=(size_t)((intersect.start_time-time_start)*rate);
                    size_t end_index=(size_t)((intersect.end_time-time_start)*rate);
                    for(size_t i=std::max(0ul,start_index);i<end_index;i++){

                        mask[i]=1;
                    }
                }
            }
        }
        

        std::unordered_map<std::string,torch::Tensor> operator[](size_t index){

            std::tuple<std::string,size_t,size_t,size_t> chunk=chunk_indices[index];
            std::pair<std::vector<std::vector<float>>,int> wav=kaldi_data::loadWav(kaldi_obj.wavs[std::get<0>(chunk)],std::get<2>(chunk)*rate,std::get<3>(chunk)*rate);
            std::vector<float> data_signal=wav.first[std::get<1>(chunk)];//data from appropriate channel

            //Adding noise
            std::string noise_file=noise_files[rand()%noise_files.size()];
            std::pair<std::vector<std::vector<float>>,int> noise_wav=kaldi_data::loadWav(noise_file);
            if(noise_wav.second!=rate){
                throw std::runtime_error("Noise and signal have different sampling rate");
            }
            std::vector<float> clean_rms;
            for(size_t i=0;i<wav.first.size();i++){
                clean_rms.push_back(features::rms(wav.first[i]));
            }
            if(wav.first[0].size() > noise_wav.first[0].size()){
                int ratio = std::ceil(static_cast<double>(wav.first[0].size()) / noise_wav.first[0].size());
                std::vector<float> temp_noise = noise_wav.first[0];
                for(int i = 0; i < ratio; i++){
                    noise_wav.first[0].insert(noise_wav.first[0].end(), temp_noise.begin(), temp_noise.end());
                }
            }
            if(wav.first[0].size() < noise_wav.first[0].size()){
                int start = 0;
                noise_wav.first[0] = std::vector<float>(noise_wav.first[0].begin() + start, noise_wav.first[0].begin() + start + wav.first[0].size());
            }
            std::vector<float> noise_rms;
            for(size_t i=0;i<wav.first.size();i++){
                noise_rms.push_back(features::rms(noise_wav.first[i]));
            }
            float snr_db=snr[rand()%snr.size()];//Choosing a sound to noise ratio
            //updating the rms for the noise
            std::vector<float> noise_rms_updated;
            for(size_t i=0;i<wav.first.size();i++){
                noise_rms_updated.push_back(std::sqrt(clean_rms[i]*clean_rms[i]/std::pow(10,snr_db/10)));
            }
            //scaling the noise
            for(size_t i=0;i<wav.first.size();i++){
                for(size_t j=0;j<noise_wav.first[i].size();j++){
                    noise_wav.first[i][j]*=noise_rms_updated[i]/noise_rms[i];
                }
            }
            //adding noise to the signal
            for(size_t i=0;i<wav.first.size();i++){
                for(size_t j=0;j<wav.first[i].size();j++){
                    wav.first[i][j]+=noise_wav.first[i][j];
                }
            }


            //Normalisation
            //did not normalise look into it


            torch::Tensor mixed_signal_tensor = torch::from_blob(wav.first.data(), {static_cast<int64_t>(wav.first.size())}).clone();
            std::vector<int> mask=getMaskFromRTTM(std::get<0>(chunk),data_signal.size(),std::get<2>(chunk),std::get<3>(chunk));
            torch::Tensor mask_tensor = torch::from_blob(mask.data(), {static_cast<int64_t>(mask.size())}).clone();\
            return {{"feat",mixed_signal_tensor},{"mask",mask_tensor}};
        }
        std::optional<size_t> size() const {
            return total_chunk;
        }
    };
}
#include <torch/torch.h>
#include <string>
#include "../../../utils/kaldi_data.cpp"
#include "../utils_vad/features.cpp"
#include <vector>
#include <tuple>
#include <map>
#include <iostream>
#include <algorithm>

namespace dataLoader{
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


        std::string data_path,rttm_path;
        float chunk_size,chunk_shift;
        int rate;
        size_t total_chunk;
        std::vector<std::tuple<std::string,size_t,size_t,size_t>> chunk_indices;
        kaldi_data::KaldiData kaldi_obj;

        Dataset(std::string data_path,std::string rttm_path,float chunk_size=5,float chunk_shift=4,int rate=16000):
        data_path(data_path),rttm_path(rttm_path),chunk_size(chunk_size),chunk_shift(chunk_shift),rate(rate),kaldi_obj(kaldi_data::KaldiData(data_path)){
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
            std::vector<float> data_signal=wav.first[std::get<1>(chunk)];
            torch::Tensor data_signal_tensor = torch::from_blob(data_signal.data(), {static_cast<int64_t>(data_signal.size())}).clone();
            std::vector<int> mask=getMaskFromRTTM(std::get<0>(chunk),data_signal.size(),std::get<2>(chunk),std::get<3>(chunk));
            torch::Tensor mask_tensor = torch::from_blob(mask.data(), {static_cast<int64_t>(mask.size())}).clone();\
            return {{"data_signal",data_signal_tensor},{"mask",mask_tensor}};
        }
        std::optional<size_t> size() const {
            return total_chunk;
        }
    };
}
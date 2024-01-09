#include <onnxruntime/core/session/onnxruntime_cxx_api.h>
#include <iostream>
#include <vector>
#include <string>
#include "../../utils/kaldi_data.cpp"
#include <sndfile.h>
#include <torch/torch.h>
#include <fstream>
#include <filesystem>
#include <chrono>
#include <set>

int inference_length=13001600;

double evaluation_single_file(const std::string& output_file,const std::string& rttm_file){
    std::ifstream output(output_file);
    std::ifstream rttm(rttm_file);
    std::string line;
    std::vector<std::pair<double,double>> output_set;
    std::vector<std::pair<double,double>> rttm_set;
    while(std::getline(output,line)){
        double start,end;
        sscanf(line.c_str(),"Start: %lf End: %lf",&start,&end);
        output_set.push_back({start,end});
    }
    while(std::getline(rttm,line)){
        double start,end;
        sscanf(line.c_str(),"SPEAKER 1 1 %lf %lf",&start,&end);
        rttm_set.push_back({start,end});
    }
    double tper = 0;
    double last_end = 0;
    for(int i = 0, j = 0; i < rttm_set.size() && j < output_set.size();) {
        if(output_set[j].first >= rttm_set[i].first && output_set[j].second <= rttm_set[i].second) {
            tper += output_set[j].second - output_set[j].first;
            last_end = output_set[j].second;
            j++;
        } else if(output_set[j].first < rttm_set[i].first) {
            if(output_set[j].second > rttm_set[i].first) {
                tper += rttm_set[i].second - rttm_set[i].first;
                last_end = rttm_set[i].second;
                i++;
            } else {
                tper += output_set[j].first - last_end;
                last_end = output_set[j].second;
                j++;
            }
        } else {
            tper += rttm_set[i].first - last_end;
            last_end = rttm_set[i].second;
            i++;
        }
    }
    std::cout<<"Percentage of time of correct detection:"<<100*tper/rttm_set.back().second<<std::endl;
    double tp=0,fp=0,fn=0;
    for(const auto& x:output_set){
        bool flag=false;
        for(const auto& y:rttm_set){
            if(x.first>=y.first&&x.second<=y.second){
                tp++;
                flag=true;
                break;
            }
        }
        if(!flag){
            fp++;
        }
    }
    for(const auto& x:rttm_set){
        bool flag=false;
        for(const auto& y:output_set){
            if(x.first>=y.first&&x.second<=y.second){
                flag=true;
                break;
            }
        }
        if(!flag){
            fn++;
        }
    }
    double precision=tp/(tp+fp);
    double recall=tp/(tp+fn);
    double f1=2*precision*recall/(precision+recall);
    std::cout<<"Precision: "<<precision<<std::endl;
    std::cout<<"Recall: "<<recall<<std::endl;
    std::cout<<"F1: "<<f1<<std::endl;
    return f1;
}

double getDuration(const std::string& filename) {
    SF_INFO sndInfo;
    SNDFILE *sndFile = sf_open(filename.c_str(), SFM_READ, &sndInfo);
    if (sndFile == NULL) {
        std::cerr << "Error reading source file" << std::endl;
        return -1;
    }

    double duration = static_cast<double>(sndInfo.frames) / sndInfo.samplerate;
    sf_close(sndFile);
    return duration;
}
torch::Tensor convertVecVecToTensor(std::vector<std::vector<float>>& vec,bool freeMemory=false) {
    // Flatten the 2D vector into a 1D vector
    std::vector<float> flatVec;
    for (const auto& subVec : vec) {
        for(const auto& val : subVec) {
            flatVec.push_back(val);
        }
    }

    // Create a 1D tensor from the 1D vector
    torch::Tensor tensor = torch::tensor(flatVec);

    // Reshape the 1D tensor into a 2D tensor
    tensor = tensor.view({static_cast<int64_t>(vec.size()), static_cast<int64_t>(vec[0].size())});
    if(freeMemory){
        vec.clear();
    }
    return tensor;
}
std::vector<std::pair<float,float>> inference(std::vector<float>& audio_data,Ort::Session& model,const int chunk_size,const int sampling_rate,const float threshold,const double duration_seconds,const double initial_duration_seconds=0){
    std::vector<int64_t> input_tensor_shape = {1,1, static_cast<int64_t>(audio_data.size())/*chunk_sample*/}; // shape for 1D tensor
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    auto input_tensor = Ort::Value::CreateTensor<float>(memory_info, audio_data.data(), audio_data.size(), input_tensor_shape.data(), input_tensor_shape.size());

    // Score model & input tensor, get back output tensor
    std::vector<const char*> input_node_names = {"input"}; // replace with your input node name
    std::vector<const char*> output_node_names = {"output"}; // replace with your output node name
    auto output_tensors = model.Run(Ort::RunOptions{nullptr}, input_node_names.data(), &input_tensor, 1, output_node_names.data(), 1);

    // Get pointer to output tensor float values
    float* floatarr = output_tensors.front().GetTensorMutableData<float>();

    // Get the shape of the output tensor
    std::vector<int64_t> output_shape = output_tensors.front().GetTensorTypeAndShapeInfo().GetShape();

    // Calculate the size of the output tensor
    int64_t output_size = std::accumulate(output_shape.begin(), output_shape.end(), 1, std::multiplies<int64_t>());
    double time_per_frame=1.0*duration_seconds/output_size;


    // Print the output tensor data
    std::vector<std::pair<float,float>>output_pairs;
    bool flag=false;
    double out=0;
    for (int64_t i = 0; i < output_size; i++) {
        if(floatarr[i]>threshold){
            floatarr[i]=1;
        }
        else{
            floatarr[i]=0;
        }
        double pres_time=i*time_per_frame;
        if(time_per_frame>duration_seconds){
            break;
        }
        if(floatarr[i]==1){
            if(!flag){
                flag=true;
                out=pres_time;
            }
        }
        else{
            if(flag){
                flag=false;
                output_pairs.push_back({ initial_duration_seconds+out,initial_duration_seconds+std::min(pres_time+time_per_frame,duration_seconds)});
            }
        }
    }
    if(flag){
        output_pairs.push_back({initial_duration_seconds+out,initial_duration_seconds+duration_seconds});
    }
    return output_pairs;
}

void inference_single(const std::string& input_file,Ort::Session& model,const std::string& output_file){
    //Initailization
    float stride_ms=80,chunk_ms=100,sampling_rate=16000,threshold=0.7,min_speech_sec=0.5;
    int chunk_sample = int(chunk_ms * sampling_rate / 1000),stride_sample = int(stride_ms * sampling_rate / 1000);
    
    //wav loading
    auto wav=kaldi_data::loadWav(input_file);
    size_t len_audio=wav.first[0].size();
    int len_pad=chunk_sample-len_audio%chunk_sample;
    std::vector<float> audio_data=wav.first[0];
    audio_data.insert(audio_data.end(),len_pad,0);
    if(wav.second!=sampling_rate){
        throw std::runtime_error("Sampling rate of wav file is not equal to "+std::to_string(sampling_rate));
    }

    //time loading
    double duration_seconds=getDuration(input_file);

    // inference
    // Convert your 1D audio tensor to a std::vector<float>
    std::vector<std::pair<float,float>> output_pairs;
    for(int i=0;i<len_audio;i+=stride_sample){
        std::vector<float> audio_data_chunk(audio_data.begin()+i,audio_data.begin()+i+chunk_sample);
        auto temp=inference(audio_data_chunk,model,chunk_sample,sampling_rate,threshold,duration_seconds*chunk_sample/len_audio,duration_seconds*i/len_audio);
        for(auto x:temp){
            if(output_pairs.size()!=0&&x.first<=output_pairs.back().second){
                output_pairs.back().second=std::max(output_pairs.back().second,x.second);
            }
            else{
                output_pairs.push_back(x);
            }
        }
    }
    std::ofstream print_output(output_file);
    for(auto x:output_pairs){
        if(x.second-x.first>min_speech_sec)
        print_output<<"Start: "<<x.first<<" End: "<<x.second<<std::endl;
    }
    print_output.close();
}
void inference_folder(const std::string& input_folder,Ort::Session& model,const std::string& output_folder){
    std::vector<std::string> files;
    for (const auto & entry : std::filesystem::directory_iterator(input_folder)) {
        if(entry.path().extension()==".flac"){
            files.push_back(entry.path().string());
        }
    }
    std::sort(files.begin(),files.end());
    int i=0,tot=files.size();
    double total_audio_time=0;
    auto start=std::chrono::high_resolution_clock::now();
    size_t m=0;
    for(const auto& file:files){
        // if(i<9){i++;continue;}
        total_audio_time+=getDuration(file);
        std::cout << "\rProcessing file number: "<<++i<<" / "<<tot<< std::flush;
        auto x=kaldi_data::loadWav(file);
        m=std::max(m,x.first[0].size());
        std::string output_file=output_folder+"/"+file.substr(file.find_last_of("/")+1,file.find_last_of(".")-file.find_last_of("/")-1)+".txt";
        inference_single(file,model,output_file);
    }
    auto stop=std::chrono::high_resolution_clock::now();
    std::cout<<"\rCompleted"<<std::endl<<"Total audio time: "<<total_audio_time<<std::endl;
    std::cout<<"Total time taken: "<<std::chrono::duration_cast<std::chrono::seconds>(stop-start).count()<<std::endl;
    std::cout<<"Max length: "<<m<<std::endl;

}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <command> [<args>]" << std::endl;
        return 1;
    }
    std::string command = argv[1];
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ONNXModelLoader"); // Initialize ONNX Runtime environment

    // Path to your ONNX model file
    const char* model_path = "/home/rohan/VAD/inference/pretrained_models/models/best_dh.onnx";

    Ort::SessionOptions session_options;
    Ort::Session model(env, model_path, session_options); // Load the ONNX model
    if(command=="single"){
        inference_single(argv[2],model,argv[4]);
    }
    else if(command=="folder"){
        inference_folder(argv[2],model,argv[4]);
    }
    else if(command=="eval"){
        evaluation_single_file(argv[2],argv[3]);
    }
    else{
        std::cerr << "Usage: " << argv[0] << " <command> [<args>]" << std::endl;
        return 1;
    }
    return 0;
}

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

void evaluation_single_file(const std::string& output,const std::string& file,const std::string& audio){
    int samplerate=16000;
    int len_audio=getDuration(audio)*samplerate+2;
    std::vector<bool> rttm(len_audio,false),output_vec(len_audio,false);
    std::ifstream rttm_file(file);
    std::string line;
    while(std::getline(rttm_file,line)){
        double start,duration;
        char speaker[50]; // Adjust size as needed
        sscanf(line.c_str(),"SPEAKER %s 1 %lf %lf", speaker, &start, &duration);
        for(int i=(int)(start*samplerate);i<=(int)((start+duration)*samplerate);i++){
            rttm[i]=true;
        }
    }
    std::ifstream output_file(output);
    while(std::getline(output_file,line)){
        double start,end;
        sscanf(line.c_str(),"Start: %lf End: %lf",&start,&end);
        for(int i=(int)(start*samplerate);i<=(int)(end*samplerate);i++){
            output_vec[i]=true;
        }
    }
    double acc,fa,miss;acc=fa=miss=0;
    for(int i=0;i<len_audio;i++){
        if((rttm[i]&&output_vec[i])||(!rttm[i]&&!output_vec[i])){
            acc++;
        }
        else if(rttm[i]&&!output_vec[i]){
            miss++;
        }
        else if(!rttm[i]&&output_vec[i]){
            fa++;
        }
    }
    std::cout<<"Accuracy: "<<acc/len_audio<<std::endl;
    std::cout<<"Miss: "<<miss/len_audio<<std::endl;
    std::cout<<"False Alarm: "<<fa/len_audio<<std::endl;
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
    float stride_ms=5000,chunk_ms=5000,sampling_rate=16000,threshold=0.5,min_speech_sec=0;
    int chunk_sample = int(chunk_ms * sampling_rate / 1000),stride_sample = int(stride_ms * sampling_rate / 1000);
    
    //wav loading
    auto wav=kaldi_data::loadWav(input_file);
    size_t len_audio=wav.first[0].size();
    int len_pad=chunk_sample-len_audio%chunk_sample;
    std::vector<float> audio_data=wav.first[0];
    audio_data.insert(audio_data.end(),len_pad,0);
    if(wav.second!=sampling_rate){
        throw std::runtime_error("Sampling rate of wav file is not equal to "+std::to_string(sampling_rate)+" and found to be "+std::to_string(wav.second));
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
    for(const auto& file:files){
        // if(i<9){i++;continue;}
        total_audio_time+=getDuration(file);
        std::cout << "\rProcessing file number: "<<++i<<" / "<<tot<< std::flush;
        auto x=kaldi_data::loadWav(file);
        std::string output_file=output_folder+"/"+file.substr(file.find_last_of("/")+1,file.find_last_of(".")-file.find_last_of("/")-1)+".txt";
        inference_single(file,model,output_file);
    }
    auto stop=std::chrono::high_resolution_clock::now();
    std::cout<<"\rCompleted"<<std::endl<<"Total audio time: "<<total_audio_time<<std::endl;

}

void test(){

}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <command> [<args>]" << std::endl;
        return 1;
    }
    std::string command = argv[1];
    auto start=std::chrono::high_resolution_clock::now();
    if(command=="single"){
        std::string arg3, arg2, arg4;
        std::ifstream file("./inference/data.txt");
        if (file.is_open()) {
            std::getline(file, arg2);
            std::getline(file, arg3);
            std::getline(file, arg4);
            file.close();
        } else {
            std::cerr << "Unable to open file data.txt";
            return 1;
        }
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ONNXModelLoader"); // Initialize ONNX Runtime environment
        // Path to your ONNX model file
        const std::string model_path =arg3;

        Ort::SessionOptions session_options;
        Ort::Session model(env, model_path.c_str(), session_options); // Load the ONNX model
    
        inference_single(arg2,model,arg4);
    }
    else if(command=="folder"){
        std::string arg3, arg2, arg4;
        std::ifstream file("./inference/data.txt");
        if (file.is_open()) {
            std::getline(file, arg2);
            std::getline(file, arg3);
            std::getline(file, arg4);
            file.close();
        } else {
            std::cerr << "Unable to open file data.txt";
            return 1;
        }
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ONNXModelLoader"); // Initialize ONNX Runtime environment
        // Path to your ONNX model file
        const char* model_path = arg3.c_str();

        Ort::SessionOptions session_options;
        Ort::Session model(env, model_path, session_options); // Load the ONNX model
    
        inference_folder(arg2,model,arg4);
    }
    else if(command=="eval"){
        std::string arg3, arg2,arg4;
        std::ifstream file("./inference/data.txt");
        if (file.is_open()) {
            std::getline(file, arg2);
            std::getline(file, arg3);
            std::getline(file, arg4);
            file.close();
        } else {
            std::cerr << "Unable to open file data.txt";
            return 1;
        }
        evaluation_single_file(arg2,arg3,arg4);//arg2: output file, arg3: rttm file, arg4: audio file
    }
    else{
        test();
    }
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);

    auto hours = duration.count() / 3600000;
    duration -= std::chrono::milliseconds(hours * 3600000);

    auto minutes = duration.count() / 60000;
    duration -= std::chrono::milliseconds(minutes * 60000);

    auto seconds = duration.count() / 1000;
    duration -= std::chrono::milliseconds(seconds * 1000);

    auto milliseconds = duration.count();

    std::cout << "Total time taken: " << hours << " hr " << minutes << " min " << seconds << " sec " << milliseconds << " msec" << std::endl;
    
    return 0;
}

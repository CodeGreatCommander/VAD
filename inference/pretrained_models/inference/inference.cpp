#include "../header_files/inference.h"

using namespace std;



std::pair<double,std::vector<float>> inference(std::vector<float>& audio_data,Ort::Session& model,const int chunk_size,const int sampling_rate,const float threshold,const double duration_seconds,const double initial_duration_seconds){
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
    std::vector<float> output(floatarr, floatarr + output_size);
    // Print the output tensor data
    // std::vector<std::pair<float,float>>output_pairs;
    // bool flag=false;
    // double out=0;
    // for (int64_t i = 0; i < output_size; i++) {
    //     file<<floatarr[i]<<" ";
    //     if(floatarr[i]>threshold){
    //         floatarr[i]=1;
    //     }
    //     else{
    //         floatarr[i]=0;
    //     }
    //     double pres_time=i*time_per_frame;
    //     if(time_per_frame>duration_seconds){
    //         break;
    //     }
    //     if(floatarr[i]==1){
    //         if(!flag){
    //             flag=true;
    //             out=pres_time;
    //         }
    //     }
    //     else{
    //         if(flag){
    //             flag=false;
    //             output_pairs.push_back({ initial_duration_seconds+out,initial_duration_seconds+std::min(pres_time/*+time_per_frame*/,duration_seconds)});
    //         }
    //     }
    // }
    // if(flag){
    //     output_pairs.push_back({initial_duration_seconds+out,initial_duration_seconds+duration_seconds});
    // }
    // file<<std::endl;
    // file.close();
    return {time_per_frame,output};
}


void inference_single(const std::string& input_file,Ort::Session& model,const std::string& output_file){
    //Initailization
    float stride_ms=100,chunk_ms=1000,sampling_rate=16000,threshold=0.55,min_speech_sec=0.2;
    int chunk_sample = int(chunk_ms * sampling_rate / 1000),stride_sample = int(stride_ms * sampling_rate / 1000);
    
    //wav loading
    auto wav=kaldi_data::loadWav(input_file);
    size_t len_audio=wav.first[0].size();
    size_t len_pad=chunk_sample-len_audio%chunk_sample;
    len_audio+=len_pad;
    std::vector<float> audio_data=wav.first[0];
    audio_data.insert(audio_data.end(),len_pad,0);
    if(wav.second!=sampling_rate){
        throw std::runtime_error("Sampling rate of wav file is not equal to "+std::to_string(sampling_rate)+" and found to be "+std::to_string(wav.second));
    }
    // std::ofstream view_file("view.txt", std::ios::app);
    // if (!view_file.is_open()) {
    //     std::cerr << "Unable to open file view.txt";
    // }
    // for (float sample : audio_data) {
    //     view_file << sample << " ";
    // }
    // view_file << std::endl;
    // view_file.close();
    //time loading
    double duration_seconds=getDuration(input_file);
    duration_seconds=duration_seconds+1.0*len_pad/sampling_rate;
    // inference
    // Convert your 1D audio tensor to a std::vector<float>
    std::vector<std::vector<float>> output_chunks;
    double time_per_frame;
    for(int i=0;i<len_audio;i+=stride_sample){
        std::vector<float> audio_data_chunk(audio_data.begin()+i,audio_data.begin()+i+chunk_sample);
        auto temp=inference(audio_data_chunk,model,chunk_sample,sampling_rate,threshold,duration_seconds*chunk_sample/len_audio,duration_seconds*i/len_audio);
        time_per_frame=temp.first;
        output_chunks.push_back(temp.second);
    }

    //length of one output term is time_per_frame in case of a term not residing completely within it the previous overlap region is considered
    int len_output=std::ceil(duration_seconds/time_per_frame*2);
    std::vector<float> output(len_output,0),count(len_output,0);
    for(int i=0;i<output_chunks.size();i++){
        for(int j=0;j<output_chunks[i].size();j++){
            double time=i*stride_ms/1000+j*time_per_frame;
            int index=time/time_per_frame;
            if(index>=len_output){
                throw std::runtime_error("Index out of bound");
            }
            double weight=normal(j,output_chunks[i].size()-j);
            output[index]+=weight*output_chunks[i][j];
            count[index]+=weight;
        }
    }
    for(int i=0;i<output.size();i++){
        if(count[i]!=0){
            output[i]/=count[i];
        }
        if(output[i]>threshold){
            output[i]=1;
        }
        else{
            output[i]=0;
        }
    }
    bool flag=false;
    double out=0;
    std::ofstream file(output_file,std::ios::trunc);
    if (!file.is_open()) {
        std::cerr << "Unable to open file "<<output_file;
    }
    for (int64_t i = 0; i < output.size(); i++) {
        if(output[i]==1){
            if(!flag){
                flag=true;
                out=i*time_per_frame;
            }
        }
        else{
            if(flag){
                flag=false;
                if(i*time_per_frame-out>min_speech_sec)
                file<<"Start: "<<out<<" End: "<<std::min(i*time_per_frame,duration_seconds)<<std::endl;
            }
        }
    }
    if(flag){
        file<<"Start: "<<out<<" End: "<<duration_seconds<<std::endl;
    }
    file.close();
}
void inference_batch(const std::string& input_folder,Ort::Session& model,const std::string& output_folder){
    std::vector<std::string> files;
    for (const auto & entry : std::filesystem::directory_iterator(input_folder)) {
        if(entry.path().extension()==".flac"){
            files.push_back(entry.path().string());
        }
    }
    std::sort(files.begin(),files.end());
    int i=0,tot=files.size();
    double total_audio_time=0;
    #pragma omp parallel for reduction(+:total_audio_time)
    for(int i = 0; i < files.size(); i++) {
        const auto& file = files[i];
        total_audio_time += getDuration(file);
        std::cout << "\rProcessing file number: " << i+1 << " / " << tot << std::flush;
        auto x = kaldi_data::loadWav(file);
        std::string output_file = output_folder + "/" + file.substr(file.find_last_of("/") + 1, file.find_last_of(".") - file.find_last_of("/") - 1) + ".txt";
        inference_single(file, model, output_file);
    }
    auto stop=std::chrono::high_resolution_clock::now();
    std::cout<<"\rCompleted"<<std::endl<<"Total audio time: "<<total_audio_time<<std::endl;

}


void infer(const std::string& input,const std::string& model_type,bool batch){
    map<string,string> model_map;
    model_map["pyannote"]="./models/best_dh.onnx";
    model_map["silero"]="./models/silero.onnx";
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ONNXModelLoader"); 
    const std::string model_path = "./models/best_dh.onnx";
    Ort::SessionOptions session_options;
    Ort::Session model(env, model_path.c_str(), session_options); // Load the ONNX model
    if(batch){
        inference_batch(input,model,"./output");
    }
    else{
        inference_single(input,model,"./output/output.txt");
    }
}

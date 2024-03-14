#include "../header_files/inference.h"
#include "../header_files/silero_vad.h"

using namespace std;

silero siler;

void initialise_silero(){
    siler.size_hc = 2 * 1 * 64; // It's FIXED.
    siler._h=std::vector<float>(siler.size_hc,0);
    siler._c=std::vector<float>(siler.size_hc,0);
    siler.input_node_dims[0] = 1;
    siler.sr={16000};
}

std::pair<double,std::vector<float>> inference_silero(std::vector<float>& audio_data,Ort::Session& model,const int chunk_size,const int sampling_rate,const float threshold,const double duration_seconds,const double initial_duration_seconds){
    std::vector<int64_t> input_tensor_shape = {1,1, static_cast<int64_t>(audio_data.size())/*chunk_sample*/}; // shape for 1D tensor
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    // auto input_tensor = Ort::Value::CreateTensor<float>(memory_info, audio_data.data(), audio_data.size(), input_tensor_shape.data(), input_tensor_shape.size());
    siler.input_node_dims[1] = audio_data.size();
        Ort::Value input_ort = Ort::Value::CreateTensor<float>(
            memory_info, audio_data.data(), audio_data.size(), siler.input_node_dims, 2);
        Ort::Value sr_ort = Ort::Value::CreateTensor<int64_t>(
            memory_info, siler.sr.data(), siler.sr.size(), siler.sr_node_dims, 1);
        Ort::Value h_ort = Ort::Value::CreateTensor<float>(
            memory_info, siler._h.data(), siler._h.size(), siler.hc_node_dims, 3);
        Ort::Value c_ort = Ort::Value::CreateTensor<float>(
            memory_info, siler._c.data(), siler._c.size(), siler.hc_node_dims, 3);

        // Clear and add inputs
        siler.ort_inputs.clear();
        siler.ort_inputs.emplace_back(std::move(input_ort));
        siler.ort_inputs.emplace_back(std::move(sr_ort));
        siler.ort_inputs.emplace_back(std::move(h_ort));
        siler.ort_inputs.emplace_back(std::move(c_ort));







    // Score model & input tensor, get back output tensor
    std::vector<const char*> input_node_names = {"input", "sr", "h", "c"}; // replace with your input node name
    std::vector<const char*> output_node_names = {"output", "hn", "cn"}; // replace with your output node name
    

    auto ort_outputs = model.Run(
            Ort::RunOptions{nullptr},
            input_node_names.data(), siler.ort_inputs.data(), siler.ort_inputs.size(),
            output_node_names.data(), output_node_names.size());

    // Get pointer to output tensor float values
    float* floatarr = ort_outputs[0].GetTensorMutableData<float>();
    float *hn = ort_outputs[1].GetTensorMutableData<float>();
        std::memcpy(siler._h.data(), hn, siler.size_hc * sizeof(float));
    float *cn = ort_outputs[2].GetTensorMutableData<float>();
        std::memcpy(siler._c.data(), cn, siler.size_hc * sizeof(float));

    // Get the shape of the output tensor
    std::vector<int64_t> output_shape = ort_outputs[0].GetTensorTypeAndShapeInfo().GetShape();

    // Calculate the size of the output tensor
    int64_t output_size = std::accumulate(output_shape.begin(), output_shape.end(), 1, std::multiplies<int64_t>());
    double time_per_frame=1.0*duration_seconds/output_size;
    std::vector<float> output(floatarr, floatarr + output_size);

    return {time_per_frame,output};
}

std::pair<double,std::vector<float>> inference_pyannote(std::vector<float>& audio_data,Ort::Session& model,const int chunk_size,const int sampling_rate,const float threshold,const double duration_seconds,const double initial_duration_seconds){
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

    return {time_per_frame,output};
}


void inference_single(const std::string& input_file,Ort::Session& model,const std::string& output_file,const std::string& model_type){
    //Initailization
    initialise_silero();
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
    //time loading
    double duration_seconds=getDuration(input_file);
    duration_seconds=duration_seconds+1.0*len_pad/sampling_rate;
    // inference
    // Convert your 1D audio tensor to a std::vector<float>
    std::vector<std::vector<float>> output_chunks;
    double time_per_frame;
    for(int i=0;i<len_audio;i+=stride_sample){
        std::vector<float> audio_data_chunk(audio_data.begin()+i,audio_data.begin()+i+chunk_sample);
        std::pair<double,std::vector<float>> temp;
        if(model_type=="pyannote")
            temp=inference_pyannote(audio_data_chunk,model,chunk_sample,sampling_rate,threshold,duration_seconds*chunk_sample/len_audio,duration_seconds*i/len_audio);
        else if(model_type=="silero")
            temp=inference_silero(audio_data_chunk,model,chunk_sample,sampling_rate,threshold,duration_seconds*chunk_sample/len_audio,duration_seconds*i/len_audio);
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
void inference_batch(const std::string& input_folder,Ort::Session& model,const std::string& output_folder,const std::string& model_type){
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
        inference_single(file, model, output_file,model_type);
    }
    auto stop=std::chrono::high_resolution_clock::now();
    std::cout<<"\rCompleted"<<std::endl<<"Total audio time: "<<total_audio_time<<std::endl;

}


void infer(const std::string& input,const std::string& model_type,bool batch){
    map<string,string> model_map;
    model_map["pyannote"]="./models/best_dh.onnx";
    model_map["silero"]="./models/silero_vad.onnx";
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ONNXModelLoader"); 
    if(model_map.find(model_type)==model_map.end()){
        throw std::runtime_error("Model name not found");
    }
    const std::string model_path = model_map[model_type];
    Ort::SessionOptions session_options;
    Ort::Session model(env, model_path.c_str(), session_options); // Load the ONNX model
    if(batch){
        inference_batch(input,model,"./output",model_type);
    }
    else{
        inference_single(input,model,"./output/output.txt",model_type);
    }
}

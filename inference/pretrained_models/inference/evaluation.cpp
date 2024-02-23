#include "../header_files/evaluation.h"

using namespace std;

void evaluation_single_file(const std::string output,const std::string file,const std::string audio,int samplerate){
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
    rttm_file.close();
    std::ifstream output_file(output);
    while(std::getline(output_file,line)){
        double start,end;
        sscanf(line.c_str(),"Start: %lf End: %lf",&start,&end);
        for(int i=(int)(start*samplerate);i<=(int)(end*samplerate);i++){
            output_vec[i]=true;
        }
    }
    output_file.close();
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


void evaluate_folder(const std::string& output_folder,const std::string& rttm_folder,const std::string& audio_folder){
    std::vector<std::string> files;
    for (const auto & entry : std::filesystem::directory_iterator(output_folder)) {
        if(entry.path().extension()==".txt"){
            files.push_back(entry.path().string());
        }
    }
    std::sort(files.begin(),files.end());
    for(int i=0;i<10;i++){
        std::cout<<files[i]<<std::endl;
    }
    int i=0,tot=files.size();
    double total_audio_time=0;
    int c=0;
    for(const auto& file:files){
        c++;
        if(c==3)continue;
        std::cout<<"Started"<<std::endl;
        std::cout << "Processing file number: "<<++i<<" / "<<tot<<" "<<file<<std::endl;
        std::string rttm_file=rttm_folder+"/"+file.substr(file.find_last_of("/")+1,file.find_last_of(".")-file.find_last_of("/")-1)+".rttm";
        std::string audio_file=audio_folder+"/"+file.substr(file.find_last_of("/")+1,file.find_last_of(".")-file.find_last_of("/")-1)+".flac";
        total_audio_time+=getDuration(audio_file);
        evaluation_single_file(file,rttm_file,audio_file);
        std::cout<<"ended"<<std::endl;
    }
    auto stop=std::chrono::high_resolution_clock::now();
    std::cout<<"\rCompleted"<<std::endl<<"Total audio time: "<<total_audio_time<<std::endl;

}

void eval(const std::string& output_folder,const std::string& rttm_folder,const std::string& audio_folder){
    evaluate_folder(output_folder,rttm_folder,audio_folder);
}
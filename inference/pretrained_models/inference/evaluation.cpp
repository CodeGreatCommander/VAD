#include "../header_files/evaluation.h"

using namespace std;

pair<int,pair<int,pair<int,int>>> evaluation_single_file(const std::string output,const std::string file,const std::string audio,const std::string& response_file,int samplerate){
    try{
        int len_audio=getDuration(audio)*samplerate+2;
        std::vector<bool> rttm(len_audio,false),output_vec(len_audio,false);
        std::ifstream rttm_file(file);
        std::string line;
        while(std::getline(rttm_file,line)){
            double start,duration;
            char speaker[1000]; // Adjust size as needed
            sscanf(line.c_str(),"SPEAKER %s 1 %lf %lf", speaker, &start, &duration);
            for(int i=(int)(start*samplerate);i<=min((int)((start+duration)*samplerate),len_audio);i++){
                rttm[i]=true;
            }
        }
        rttm_file.close();
        std::ifstream output_file(output);
        while(std::getline(output_file,line)){
            double start,end;
            sscanf(line.c_str(),"Start: %lf End: %lf",&start,&end);
            for(int i=(int)(start*samplerate);i<=min((int)(end*samplerate),len_audio);i++){
                output_vec[i]=true;
            }
        }
        output_file.close();
        int acc,fa,miss;acc=fa=miss=0;
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
        std::ofstream analysis(response_file,std::ios::app);
        if (!analysis) {
            std::cerr << "Unable to open file for writing";
            return {0,{0,{0,0}}};
        }
        analysis << "File: " << file << std::endl;\
        analysis << "Length of audio: " << 1.0*len_audio/samplerate << std::endl;
        analysis << "Accuracy: " << 1.0*acc/len_audio << std::endl;
        analysis << "Miss: " << 1.0*miss/len_audio << std::endl;
        analysis << "False Alarm: " << 1.0*fa/len_audio << std::endl;
        analysis << "-----------------------------------" << std::endl;
        analysis.close();
        return {len_audio,{acc,{miss,fa}}};
    }
    catch(const std::exception& e){
        std::ofstream analysis(response_file,std::ios::app);
        if (!analysis) {
            std::cerr << "Unable to open file for writing";
            return {0,{0,{0,0}}};
        }
        analysis <<"File: "<<file<<std::endl;
        analysis << "Error: " << e.what() << std::endl;
        analysis << "-----------------------------------" << std::endl;  
        analysis.close();
        return {0,{0,{0,0}}}; 
    }
}


void evaluate_folder(const std::string& output_folder,const std::string& rttm_folder,const std::string& audio_folder,const std::string& response_file){
    std::vector<std::string> files;
    long long int len_audio=0,acc=0,miss=0,fa=0;
    for (const auto & entry : std::filesystem::directory_iterator(output_folder)) {
        if(entry.path().extension()==".txt"){
            files.push_back(entry.path().string());
        }
    }
    std::sort(files.begin(),files.end());
    int i=0,tot=files.size();
    double total_audio_time=0;
    int c=0;
    for(auto file:files){
        c++;
        std::cout << "\rProcessing file number: "<<++i<<" / "<<tot;
        std::cout.flush();
        std::string rttm_file=rttm_folder+"/"+file.substr(file.find_last_of("/")+1,file.find_last_of(".")-file.find_last_of("/")-1)+".rttm";
        std::string audio_file=audio_folder+"/"+file.substr(file.find_last_of("/")+1,file.find_last_of(".")-file.find_last_of("/")-1)+".flac";
        total_audio_time+=getDuration(audio_file);
        auto x=evaluation_single_file(file,rttm_file,audio_file,response_file);
        len_audio+=x.first;
        acc+=x.second.first;
        miss+=x.second.second.first;
        fa+=x.second.second.second;
    }
    std::ofstream analysis(response_file,std::ios::app);
    if (!analysis) {
        std::cerr << "Unable to open file for writing";
        return;
    }
    analysis << "Combined Analysis" << std::endl;
    analysis << "Total Files: " << c << std::endl;
    analysis << "Total audio time: " << total_audio_time << std::endl;
    analysis << "Accuracy: " << 1.0*acc/len_audio << std::endl;
    analysis << "Miss: " << 1.0*miss/len_audio << std::endl;
    analysis << "False Alarm: " << 1.0*fa/len_audio << std::endl;
    analysis << "-----------------------------------" << std::endl;
    analysis.close();
    std::cout<<"\rCompleted"<<std::endl<<"Total audio time: "<<total_audio_time<<std::endl;

}

void eval(const std::string& output_folder,const std::string& rttm_folder,const std::string& audio_folder,bool batch){
    if(batch){
        string filename="analysis/analysis.txt";
        std::ofstream file(filename);
        if (!file) {
            std::cerr << "Unable to open file for writing";
            return;
        }
        file.close();
        evaluate_folder(output_folder,rttm_folder,audio_folder,filename);
    }
    else{
        string filename="analysis/analysis.txt";
        std::ofstream file(filename);
        if (!file) {
            std::cerr << "Unable to open file for writing";
            return;
        }
        file.close();
        evaluation_single_file(output_folder,rttm_folder,audio_folder,filename);
    }
}
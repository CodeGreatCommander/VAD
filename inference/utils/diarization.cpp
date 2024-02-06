#include <vector>
#include <string>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <sstream>
#include <map>
namespace diarization{
    std::vector<std::string> read_rttm(std::string rttm_path){
        std::vector<std::string> rttm;
        std::ifstream rttm_file(rttm_path);
        std::string line;
        while(std::getline(rttm_file, line)){
            rttm.push_back(line);
        }
        return rttm;
    }
    void write_der_file(std::string ref_rttm,std::vector<std::string> der,std::string out_der_file){
        std::vector<std::string> rttm=read_rttm(ref_rttm),spkr_info;
        std::copy_if(rttm.begin(),rttm.end(),std::back_inserter(spkr_info),[](std::string line){return line.compare(0, 9, "SPKR-INFO") == 0; });
        std::map<std::string,bool> rec_id_list;
        size_t count=0;
        std::ofstream f(out_der_file);

        for(std::string rttm_line:rttm){
            std::istringstream iss(rttm_line);
            std::string rec_id;
            std::getline(iss, rec_id, ' '); // Skip the first element
            std::getline(iss, rec_id, ' ');
            if(!rec_id_list[rec_id]){
                rec_id_list[rec_id]=true;
                std::ostringstream ss;
                ss << rec_id << " " << std::fixed << std::setprecision(2) << der[count];
                std::string line_str = ss.str();
                f << line_str << "\n";
                count++;
            }
        }
    }
}
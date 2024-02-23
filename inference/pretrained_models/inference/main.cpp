#include "../header_files/inference.h"
#include "../header_files/evaluation.h"
#include <string>
#include <iostream>
#include <chrono>
using namespace std;


int main(int argc,char *argv[]){
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <command> [<args>]" << std::endl<<"Commands: single, batch, evaluate" << std::endl;
        return 1;
    }
    std::string command = argv[1];
    auto start=std::chrono::high_resolution_clock::now();
    if(command=="single"){
        //Arguments <path to file> <model name>
        if(argc<4){
            std::cerr << "Usage: " << argv[0] << " single <path to file> <model name>" << std::endl<<"possible model names:pyannote" << std::endl;
            return 1;
        }
        infer(argv[2],false);
    }
    else if(command=="batch"){
        //Arguments <path to file> <model name>
        if(argc<4){
            std::cerr << "Usage: " << argv[0] << " batch <path to file> <model name>" << std::endl<<"possible model names:pyannote" << std::endl;
            return 1;
        }
        infer(argv[2],true);
    }
    else if(command=="evaluate"){
        //Arguments <path to file> <path to ground truth> <path to audio>
        if(argc<5){
            std::cerr << "Usage: " << argv[0] << " evaluate <path to file> <path to ground truth> <path to audio>" << std::endl;
            return 1;
        }
        eval(argv[2],argv[3],argv[4]);
    }
    else{
        std::cerr << "Usage: " << argv[0] << " <command> [<args>]" << std::endl;
        return 1;
    }
    auto end=std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    auto hours = duration.count() / 3600000;
    duration -= std::chrono::milliseconds(hours * 3600000);
    auto minutes = duration.count() / 60000;
    duration -= std::chrono::milliseconds(minutes * 60000);
    auto seconds = duration.count() / 1000;
    duration -= std::chrono::milliseconds(seconds * 1000);
    auto milliseconds = duration.count();

    std::cout << "Time taken: " 
              << hours << "hr " 
              << minutes << "min " 
              << seconds << "sec " 
              << milliseconds << "ms" 
              << std::endl;
}
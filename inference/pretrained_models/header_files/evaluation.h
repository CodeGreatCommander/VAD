#pragma once

#include <string>
#include <iostream>
#include <vector>
#include <fstream>
#include <chrono>
#include <filesystem>
#include <algorithm>
#include "../header_files/utils.h"

using namespace std;

pair<int,pair<int,pair<int,int>>> evaluation_single_file(const std::string output,const std::string file,const std::string audio,const std::string& response_file,int samplerate=100);

void evaluate_folder(const std::string& output_folder,const std::string& rttm_folder,const std::string& audio_folder,const std::string& response_file);

void eval(const std::string& output_folder,const std::string& rttm_folder,const std::string& audio_folder,bool batch=false);
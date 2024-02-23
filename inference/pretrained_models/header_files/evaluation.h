#pragma once

#include <string>
#include <iostream>
#include <vector>
#include <fstream>
#include <chrono>
#include <filesystem>
#include <algorithm>
#include "../header_files/utils.h"

void evaluation_single_file(const std::string output,const std::string file,const std::string audio,int samplerate=100);

void evaluate_folder(const std::string& output_folder,const std::string& rttm_folder,const std::string& audio_folder);

void eval(const std::string& output_folder,const std::string& rttm_folder,const std::string& audio_folder);
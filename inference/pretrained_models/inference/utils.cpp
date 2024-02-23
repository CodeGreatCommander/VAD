#include "../header_files/utils.h"

double getDuration(const std::string& filename) {
    SF_INFO sndInfo;
    SNDFILE *sndFile = sf_open(filename.c_str(), SFM_READ, &sndInfo);
    if (sndFile == NULL) {
        std::cerr << "Error reading source file " <<filename<< std::endl;
        return -1;
    }

    double duration = static_cast<double>(sndInfo.frames) / sndInfo.samplerate;
    sf_close(sndFile);
    return duration;
}
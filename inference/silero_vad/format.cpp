#include <iostream>
#include "./wav.h" // Assuming you have defined WavReader class

int main() {
    wav::WavReader wav_reader("/home/rohan/VAD/inference/silero_vad/trial.wav");

    std::cout << "Number of channels: " << wav_reader.num_channel() << std::endl;
    std::cout << "Sample rate: " << wav_reader.sample_rate() << std::endl;
    std::cout << "Bits per sample: " << wav_reader.bits_per_sample() << std::endl;
    std::cout << "Number of samples: " << wav_reader.num_samples() << std::endl;

    return 0;
}

all:
	g++ /home/rohan/VAD/inference/silero_vad/silero_vad.cpp -I /home/rohan/VAD/onnxruntime-linux-x64-1.12.1/include/ -L /home/rohan/VAD/onnxruntime-linux-x64-1.12.1/lib/ -lonnxruntime  -Wl,-rpath,/home/rohan/VAD/onnxruntime-linux-x64-1.12.1/lib/ -o test
	./test

compile: build
	make -C ./build

build:
	mkdir -p ./build
	cmake .. -B./build
EXEC = inference

infer:
	@if [ ! -f ./build/$(EXEC) ]; then make compile; fi
	./build/$(EXEC) single "/home/rohan/VAD/dataset/third_dihard_challenge_eval/data/flac/DH_EVAL_0004.flac" "/path/to/model" "/home/rohan/VAD/inference/pretrained_models/output/output.txt"

inferfolder:
	@if [ ! -f ./build/$(EXEC) ]; then make compile; fi
	./build/$(EXEC) folder "/home/rohan/VAD/dataset/third_dihard_challenge_eval/data/flac/" "/path/to/model" "/home/rohan/VAD/inference/pretrained_models/output"

evaluate:
	@if [ ! -f ./build/$(EXEC) ]; then make compile; fi
	./build/$(EXEC) eval "/home/rohan/VAD/inference/pretrained_models/output/output.txt" "/home/rohan/VAD/dataset/third_dihard_challenge_eval/data/rttm/DH_EVAL_0004.rttm"
clear_output:
	rm -f /home/rohan/VAD/inference/pretrained_models/output/*
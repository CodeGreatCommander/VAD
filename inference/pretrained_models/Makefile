initialize:all_dependencies compile

#----------------------------------------compilation and build----------------------------------------
compile: build libtorch onnxruntime libsndfile
	make -C ./build

build:
	mkdir -p ./build
	cd ./build && cmake ..  && cd ..




#----------------------------------------Dependencies----------------------------------------
DEPENDENCIES_DIR = ../../dependencies
LIBTORCH_URL = https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-1.8.1%2Bcpu.zip
ONNXRUNTIME_URL = https://github.com/microsoft/onnxruntime.git
SNDFILE_URL = https://github.com/libsndfile/libsndfile.git
libtorch:
	@if [ ! -d $(DEPENDENCIES_DIR)/libtorch ]; then \
        mkdir -p $(DEPENDENCIES_DIR); \
        wget -O $(DEPENDENCIES_DIR)/libtorch.zip $(LIBTORCH_URL); \
        unzip $(DEPENDENCIES_DIR)/libtorch.zip -d $(DEPENDENCIES_DIR); \
		clear;\
		rm -r $(DEPENDENCIES_DIR)/libtorch.zip;\
		echo "Libtorch downloaded successfully!";\
    fi

onnxruntime:
	@if [ ! -d $(DEPENDENCIES_DIR)/onnxruntime ]; then \
        mkdir -p $(DEPENDENCIES_DIR)/onnxruntime; \
        # wget -O $(DEPENDENCIES_DIR)/onnxruntime.tgz $(ONNXRUNTIME_URL); \
        # tar xzvf $(DEPENDENCIES_DIR)/onnxruntime.tgz -C $(DEPENDENCIES_DIR); \
        # # clear; \
        # echo "ONNX Runtime downloaded successfully!"; \
		git clone --recursive "$(ONNXRUNTIME_URL)" "$(DEPENDENCIES_DIR)/onnxruntime"; \
        cd $(DEPENDENCIES_DIR)/onnxruntime && ./build.sh --config Release --build_shared_lib --parallel; \
        echo "ONNX Runtime downloaded and built successfully!"; \
    fi
libsndfile:
	@if [ ! -d $(DEPENDENCIES_DIR)/libsndfile ]; then \
		mkdir -p $(DEPENDENCIES_DIR)/libsndfile; \
        git clone $(SNDFILE_URL) $(DEPENDENCIES_DIR)/libsndfile; \
        cd $(DEPENDENCIES_DIR)/libsndfile && ./autogen.sh && ./configure && make && sudo make install; \
        echo "libsndfile downloaded and installed successfully!"; \
    fi
all_dependencies: libtorch onnxruntime libsndfile




#----------------------------------------run----------------------------------------
EXEC = inference

infer:
	@mkdir -p ./output
	@if [ ! -f ./build/$(EXEC) ]; then make compile; fi
	@echo "Inference started..."
	@./build/$(EXEC) single $(FILE) $(MODEL)
	@echo "Inference completed successfully!"

batch:
	@mkdir -p ./output
	@if [ ! -f ./build/$(EXEC) ]; then make compile; fi
	@echo "Batch inference started..."
	@./build/$(EXEC) batch $(DIR) $(MODEL)
	@echo "Batch inference completed successfully!"
evaluate:
	@mkdir -p ./analysis
	@if [ ! -f ./build/$(EXEC) ]; then make compile; fi
	@./build/$(EXEC) evaluate $(FILE) $(RTTM) $(AUDIO) 
evaluate_batch:
	@mkdir -p ./analysis
	@if [ ! -f ./build/$(EXEC) ]; then make compile; fi
	@./build/$(EXEC) evaluate_batch $(DIR) $(RTTM_DIR) $(AUDIO_DIR)
test:
	@if [ ! -f ./build/$(EXEC) ]; then make compile; fi
	./build/$(EXEC) test 


#---------------------------------------------clean--------------------------------------------------
make clean:clear_output clear_dependencies clear_build

clear_build:
	rm -rf ./build
clear_output:
	rm -f ./output/*
clear_dependencies:
	rm -rf $(DEPENDENCIES_DIR)/*
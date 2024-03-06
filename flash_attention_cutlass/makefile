run_alone: build_standalone
	./build/flash_attention_cutlass_standalone

build_standalone:
	cmake -B build
	cmake --build build

build:
	python build.py install

dbg: build_standalone
	cuda-gdb ./build/main

.PHONY: build


run: build
	./self_attention_standalone
	./flash_attention_v1_standalone
	./flash_attention_v2_standalone

build:
	nvcc -o self_attention_standalone self_attention_standalone.cu
	nvcc -o flash_attention_v1_standalone flash_attention_v1_standalone.cu
	nvcc -o flash_attention_v2_standalone flash_attention_v2_standalone.cu


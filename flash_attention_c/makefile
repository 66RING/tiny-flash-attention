build:
	cmake -B build  -D CMAKE_EXPORT_COMPILE_COMMANDS=ON \
		-DCMAKE_PREFIX_PATH=`python -c 'import torch;print(torch.utils.cmake_prefix_path)'` \
		-Dpybind11_DIR=`pybind11-config --cmakedir` -DCMAKE_BUILD_TYPE=DEBUG
	cmake --build build

.PHONY: build

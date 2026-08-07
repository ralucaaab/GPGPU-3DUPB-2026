NVCC ?= nvcc
GPP ?= g++
# TODO: Check the GPU used by the queue which you use on the cluster. (A100 if you use the demo one, dgxa100)
# Modify the CUDA_ARCH variable to match your GPU architecture. You can check your GPU architecture at https://developer.nvidia.com/cuda/gpus and set the appropriate value.
CUDA_ARCH ?= sm_120
NVCCFLAGS ?= -O2 -std=c++17 -arch=$(CUDA_ARCH)
CXXFLAGS ?= -O2 -std=c++17

COUNTING_GFLOPS_HEADERS := $(wildcard src/1_countingGFLOPS/*.cuh)
CUDA_PROPERTIES_HEADERS := $(wildcard src/2_cudaProperties/*.cuh)
VARIABLE_SUM_HEADERS := $(wildcard src/3_variableSum/*.cuh)
MATRIX_MUL_HEADERS := $(wildcard src/4_matrixMul/*.cuh)
RAY_TRACER_SIMPLE_SERIAL_FILES := $(wildcard src/5_rayTracerSimpleSerial/*)
RAY_TRACER_SIMPLE_CUDA_FILES := $(wildcard src/6_rayTracerSimpleCUDA/*)
RAY_TRACER_ADVANCED_CUDA_FILES := $(wildcard src/7_rayTracerAdvancedCUDA/*)

all: countingGFLOPS cudaProperties variableSum matrixMul rayTracerSimpleSerial rayTracerSimpleCUDA rayTracerAdvancedCUDA

countingGFLOPS: src/1_countingGFLOPS/CountingGFLOPS.cu $(COUNTING_GFLOPS_HEADERS)
	$(NVCC) $(NVCCFLAGS) $< -o $@

cudaProperties: src/2_cudaProperties/CudaProperties.cu $(CUDA_PROPERTIES_HEADERS)
	$(NVCC) $(NVCCFLAGS) $< -o $@

variableSum: src/3_variableSum/VariableSum.cu $(VARIABLE_SUM_HEADERS)
	$(NVCC) $(NVCCFLAGS) $< -o $@

matrixMul: src/4_matrixMul/MatrixMultiply.cu $(MATRIX_MUL_HEADERS)
	$(NVCC) $(NVCCFLAGS) $< -o $@

rayTracerSimpleSerial: src/5_rayTracerSimpleSerial/final_scene.cpp $(RAY_TRACER_SIMPLE_SERIAL_FILES)
	$(GPP) $(CXXFLAGS) $< -o $@

rayTracerSimpleCUDA: src/6_rayTracerSimpleCUDA/final_scene.cu $(RAY_TRACER_SIMPLE_CUDA_FILES)
	$(NVCC) $(NVCCFLAGS) $< -o $@

rayTracerAdvancedCUDA: src/7_rayTracerAdvancedCUDA/finalScene.cu $(RAY_TRACER_ADVANCED_CUDA_FILES)
	$(NVCC) $(NVCCFLAGS) $< -o $@


clean:
	rm -f countingGFLOPS cudaProperties variableSum matrixMul rayTracerSimpleSerial rayTracerSimpleCUDA rayTracerAdvancedCUDA

.PHONY: all clean

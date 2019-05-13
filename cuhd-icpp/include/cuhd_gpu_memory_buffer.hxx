/*****************************************************************************
 *
 * CUHD - A massively parallel Huffman decoder
 *
 * released under LGPL-3.0
 *
 * 2017-2018 André Weißenberger
 *
 *****************************************************************************/

#include "cuhd_gpu_memory_buffer.h"
#include "cuhd_cuda_definitions.h"

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cuda.h>

template<typename T>
cuhd::CUHDGPUMemoryBuffer<T>::CUHDGPUMemoryBuffer(T* buffer, size_t size)
    : buffer_(buffer),
       size_(size)  {
    is_allocated_ = false;
}

template<typename T>
cuhd::CUHDGPUMemoryBuffer<T>::~CUHDGPUMemoryBuffer() {
    free();
}

template<typename T>
T* cuhd::CUHDGPUMemoryBuffer<T>::get() {
    return buffer_device_;   
}

template<typename T>
void cuhd::CUHDGPUMemoryBuffer<T>::allocate() {
    cudaMalloc(&buffer_device_, size_ * sizeof(T));
	CUERR
    
    is_allocated_ = true;
}

template<typename T>
void cuhd::CUHDGPUMemoryBuffer<T>::free() {
    if(!is_allocated_) return;
    
    cudaFree(buffer_device_);
	CUERR
	
    is_allocated_ = false;
}

template<typename T>
void cuhd::CUHDGPUMemoryBuffer<T>::cpy_host_to_device() {
    if(!is_allocated_) allocate();

    cudaMemcpy(buffer_device_, buffer_, size_ * sizeof(T),
        cudaMemcpyHostToDevice);
    CUERR
}

template<typename T>
void cuhd::CUHDGPUMemoryBuffer<T>::cpy_device_to_host() {
    cudaMemcpy(buffer_, buffer_device_, size_ * sizeof(T),
        cudaMemcpyDeviceToHost);
    CUERR
}


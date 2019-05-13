/*****************************************************************************
 *
 * CUHD - A massively parallel Huffman decoder
 *
 * released under LGPL-3.0
 *
 * 2017-2018 André Weißenberger
 *
 *****************************************************************************/

#ifndef CUHD_GPU_MEMORY_BUFFER_
#define CUHD_GPU_MEMORY_BUFFER_

#include "cuhd_constants.h"

#include <cstdlib>
#include <memory>

namespace cuhd {
    template <typename T>
    class CUHDGPUMemoryBuffer {
        public:
            CUHDGPUMemoryBuffer(T* buffer, size_t size);
            ~CUHDGPUMemoryBuffer();
            
            T* get();
            void allocate();
            void free();
            void cpy_host_to_device();
            void cpy_device_to_host();
            
        private:
            T* buffer_;
            T* buffer_device_;

            bool is_allocated_;
            size_t size_;
    };
}

#include "cuhd_gpu_memory_buffer.hxx"

#endif /* GPU_MEMORY_BUFFER_H_ */

/*****************************************************************************
 *
 * CUHD - A massively parallel Huffman decoder
 *
 * released under LGPL-3.0
 *
 * 2017-2018 André Weißenberger
 *
 *****************************************************************************/

#include "cuhd_gpu_input_buffer.h"

cuhd::CUHDGPUInputBuffer::CUHDGPUInputBuffer(
    std::shared_ptr<CUHDInputBuffer> input_buffer)
    : CUHDGPUMemoryBuffer<UNIT_TYPE>(input_buffer->get_compressed_data(),
        input_buffer->get_compressed_size_units()),
        input_buffer_(input_buffer) {
    
}

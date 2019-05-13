/*****************************************************************************
 *
 * CUHD - A massively parallel Huffman decoder
 *
 * released under LGPL-3.0
 *
 * 2017-2018 André Weißenberger
 *
 *****************************************************************************/

#include "cuhd_gpu_output_buffer.h"

cuhd::CUHDGPUOutputBuffer::CUHDGPUOutputBuffer(
	std::shared_ptr<CUHDOutputBuffer> output_buffer)
	: CUHDGPUMemoryBuffer<SYMBOL_TYPE>(
	    output_buffer->get_decompressed_data().get(),
	    output_buffer->get_uncompressed_size()),
        output_buffer_(output_buffer) {
    
}

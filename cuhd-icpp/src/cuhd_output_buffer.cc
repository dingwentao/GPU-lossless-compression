/*****************************************************************************
 *
 * CUHD - A massively parallel Huffman decoder
 *
 * released under LGPL-3.0
 *
 * 2017-2018 André Weißenberger
 *
 *****************************************************************************/

#include "cuhd_output_buffer.h"

cuhd::CUHDOutputBuffer::CUHDOutputBuffer(size_t size) {
	uncompressed_size_ = size;

	// allocate buffer
	buffer_ = std::make_unique<SYMBOL_TYPE[]>(size);
}

std::unique_ptr<SYMBOL_TYPE[]>&
    cuhd::CUHDOutputBuffer::get_decompressed_data() {
	return buffer_;	
}

size_t cuhd::CUHDOutputBuffer::get_uncompressed_size() {
	return uncompressed_size_;
}

size_t cuhd::CUHDOutputBuffer::get_symbol_size() {
	return symbol_size_;
}

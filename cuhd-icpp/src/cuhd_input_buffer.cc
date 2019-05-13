/*****************************************************************************
 *
 * CUHD - A massively parallel Huffman decoder
 *
 * released under LGPL-3.0
 *
 * 2017-2018 André Weißenberger
 *
 *****************************************************************************/

#include "cuhd_input_buffer.h"

cuhd::CUHDInputBuffer::CUHDInputBuffer(std::uint8_t* buffer, size_t size)
    : compressed_size_(size) {

	// calculate number of units
	compressed_size_units_ = size % sizeof(UNIT_TYPE) == 0 ?
		size / sizeof(UNIT_TYPE) : size / sizeof(UNIT_TYPE) + 1;

    // avoid invalid read at end of input during decoding
    ++compressed_size_units_;
    
	// allocate buffer
	buffer_ = std::make_unique<UNIT_TYPE[]>(compressed_size_units_);
    
	// pad unused bytes at the end of the buffer with zeroes
	buffer_.get()[compressed_size_units_ - 1] = 0;

	// copy compressed data into buffer
	std::copy(buffer, buffer + size,
		reinterpret_cast<std::uint8_t*> (buffer_.get()));
}

UNIT_TYPE* cuhd::CUHDInputBuffer::get_compressed_data() {
	return buffer_.get();
}

size_t cuhd::CUHDInputBuffer::get_compressed_size() {
	return compressed_size_;
}

size_t cuhd::CUHDInputBuffer::get_compressed_size_units() {
	return compressed_size_units_;
}

size_t cuhd::CUHDInputBuffer::get_unit_size() {
	return unit_size_;	
}

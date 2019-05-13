/*****************************************************************************
 *
 * CUHD - A massively parallel Huffman decoder
 *
 * released under LGPL-3.0
 *
 * 2017-2018 André Weißenberger
 *
 *****************************************************************************/

#ifndef CUHD_INPUT_BUFFER_
#define CUHD_INPUT_BUFFER_

#include <memory>

#include "cuhd_constants.h"
#include "cuhd_definitions.h"

namespace cuhd {
    class CUHDInputBuffer {
	    public:
		    CUHDInputBuffer(std::uint8_t* buffer, size_t size);

		    // returns reference to compressed data
		    UNIT_TYPE* get_compressed_data();
		
		    size_t get_compressed_size();
		    size_t get_compressed_size_units();
		    size_t get_unit_size();

	    private:
		
		    // compressed size of input
		    size_t compressed_size_;

		    // compressed size in units
		    size_t compressed_size_units_;

		    // size of a unit
		    const size_t unit_size_ = sizeof(UNIT_TYPE);
		
		    // buffer containing the compressed input
		    cuhd_buf(UNIT_TYPE, buffer_);
    };
}
#endif /* CUHD_INPUT_BUFFER_H_ */


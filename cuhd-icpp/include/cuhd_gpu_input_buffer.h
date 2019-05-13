/*****************************************************************************
 *
 * CUHD - A massively parallel Huffman decoder
 *
 * released under LGPL-3.0
 *
 * 2017-2018 André Weißenberger
 *
 *****************************************************************************/

#ifndef CUHD_GPU_INPUT_BUFFER_
#define CUHD_GPU_INPUT_BUFFER_

#include "cuhd_gpu_memory_buffer.h"
#include "cuhd_input_buffer.h"

#include <memory>

namespace cuhd {
    class CUHDGPUInputBuffer : public CUHDGPUMemoryBuffer<UNIT_TYPE> {

	    public:
		    CUHDGPUInputBuffer(
			    std::shared_ptr<CUHDInputBuffer> input_buffer);

	    private:
		    std::shared_ptr<CUHDInputBuffer> input_buffer_;
    };
}

#endif /* CUHD_GPU_INPUT_BUFFER_H_ */


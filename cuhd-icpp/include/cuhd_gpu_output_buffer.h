/*****************************************************************************
 *
 * CUHD - A massively parallel Huffman decoder
 *
 * released under LGPL-3.0
 *
 * 2017-2018 André Weißenberger
 *
 *****************************************************************************/

#ifndef CUHD_GPU_OUTPUT_BUFFER_
#define CUHD_GPU_OUTPUT_BUFFER_

#include "cuhd_gpu_memory_buffer.h"
#include "cuhd_output_buffer.h"

#include <memory>

namespace cuhd {
    class CUHDGPUOutputBuffer : public CUHDGPUMemoryBuffer<SYMBOL_TYPE> {

	    public:
		    CUHDGPUOutputBuffer(
			    std::shared_ptr<CUHDOutputBuffer> output_buffer);

	    private:
		    std::shared_ptr<CUHDOutputBuffer> output_buffer_;
    };
}

#endif /* CUHD_GPU_OUTPUT_BUFFER_H_ */


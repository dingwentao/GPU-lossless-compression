/*****************************************************************************
 *
 * CUHD - A massively parallel Huffman decoder
 *
 * released under LGPL-3.0
 *
 * 2017-2018 André Weißenberger
 *
 *****************************************************************************/

#ifndef CUHD_GPU_CODETABLE_
#define CUHD_GPU_CODETABLE_

#include "cuhd_gpu_memory_buffer.h"
#include "cuhd_codetable.h"

namespace cuhd {
    class CUHDGPUCodetable : public CUHDGPUMemoryBuffer<CUHDCodetableItemSingle> {
        public:
            CUHDGPUCodetable(std::shared_ptr<CUHDCodetable> codetable);

            private:
                std::shared_ptr<CUHDCodetable> table_;
    };
}

#endif /* CUHD_GPU_CODETABLE_H_ */


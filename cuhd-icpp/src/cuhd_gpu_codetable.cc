/*****************************************************************************
 *
 * CUHD - A massively parallel Huffman decoder
 *
 * released under LGPL-3.0
 *
 * 2017-2018 André Weißenberger
 *
 *****************************************************************************/

#include "cuhd_gpu_codetable.h"

cuhd::CUHDGPUCodetable::CUHDGPUCodetable(
    std::shared_ptr<CUHDCodetable> codetable)
    : CUHDGPUMemoryBuffer<CUHDCodetableItemSingle>(codetable->get(),
        codetable->get_size()),
        table_(codetable) {
      
}


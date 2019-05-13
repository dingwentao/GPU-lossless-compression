/*****************************************************************************
 *
 * CUHD - A massively parallel Huffman decoder
 *
 * released under LGPL-3.0
 *
 * 2017-2018 André Weißenberger
 *
 *****************************************************************************/

#ifndef CUHD_SUBSEQUENCE_SYNC_POINT_
#define CUHD_SUBSEQUENCE_SYNC_POINT_

#include "cuhd_constants.h"

namespace cuhd {
    struct CUHDSubsequenceSyncPoint {
        std::uint32_t unit;
        std::uint32_t bit;
        std::uint32_t output_size;
        std::uint32_t synchronised;
    };
}

#endif /* CUHD_SUBSEQUENCE_SYNC_POINT_H_ */


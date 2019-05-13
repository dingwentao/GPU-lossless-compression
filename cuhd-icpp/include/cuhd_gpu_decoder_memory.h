/*****************************************************************************
 *
 * CUHD - A massively parallel Huffman decoder
 *
 * released under LGPL-3.0
 *
 * 2017-2018 André Weißenberger
 *
 *****************************************************************************/

#ifndef CUHD_GPU_DECODER_MEMORY_
#define CUHD_GPU_DECODER_MEMORY_

#include "cuhd_util.h"
#include "cuhd_constants.h"
#include "cuhd_cuda_definitions.h"
#include "cuhd_subsequence_sync_point.h"

namespace cuhd {
    class CUHDGPUDecoderMemory {
        public:
            CUHDGPUDecoderMemory(
                size_t num_units, size_t subsequence_size, size_t num_threads);
            ~CUHDGPUDecoderMemory();

            void allocate();
            void free();
            void retrieve_sync_data();

            cuhd::CUHDSubsequenceSyncPoint* get_sync_info();
            std::uint32_t* get_output_sizes();
            std::uint8_t* get_sequence_synced_device();
            std::uint8_t* get_sequence_synced_host();
            
        private:
            const size_t num_subseq_;
            const size_t num_blocks_;

            cuhd::CUHDSubsequenceSyncPoint* sync_info_;
            std::uint32_t* output_sizes_;
            std::uint8_t* sequence_synced_device_;
            std::uint8_t* sequence_synced_host_;

            bool is_allocated_;
    };
}

#endif /* CUHD_GPU_DECODER_MEMORY_H_ */


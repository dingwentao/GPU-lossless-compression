/*****************************************************************************
 *
 * CUHD - A massively parallel Huffman decoder
 *
 * released under LGPL-3.0
 *
 * 2017-2018 André Weißenberger
 *
 *****************************************************************************/

#ifndef CUHD_CODETABLE_
#define CUHD_CODETABLE_

#include "cuhd_constants.h"
#include "cuhd_definitions.h"

#include <memory>

namespace cuhd {
    struct CUHDCodetableItemSingle {
        BIT_COUNT_TYPE num_bits;
        SYMBOL_TYPE symbol;
    };

    class CUHDCodetable {
        public:
            CUHDCodetable(size_t num_entries);

            size_t get_size();
            size_t get_num_entries();
            size_t get_max_codeword_length();
            
            CUHDCodetableItemSingle* get();
            
        private:
            
            // total number of rows
            size_t size_;

            // actual number of items
            size_t num_entries_;

            cuhd_buf(CUHDCodetableItemSingle, table_);
    };
}
#endif /* CUHD_CODETABLE_H_ */


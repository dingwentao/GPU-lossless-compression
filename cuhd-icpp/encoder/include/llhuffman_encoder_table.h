/*****************************************************************************
 *
 * LLHuff - Length-limited Huffman encoding
 *
 * released under LGPL-3.0
 *
 * 2017-2018 André Weißenberger
 *
 *****************************************************************************/

#ifndef LL_HUFFMAN_ENCODER_TABLE_
#define LL_HUFFMAN_ENCODER_TABLE_

#include "cuhd_constants.h"

#include <unordered_map>

namespace llhuff {
    struct LLHuffmanEncoderTable {

        struct LLHuffmanEncoderTableItem {
            UNIT_TYPE codeword;
            size_t length;
        };
        
        // size of compressed output in units
        size_t compressed_size;

        // the dictionary
        std::unordered_map<SYMBOL_TYPE, LLHuffmanEncoderTableItem> dict;
    };
}

#endif /* LL_HUFFMAN_ENCODER_TABLE_H_ */


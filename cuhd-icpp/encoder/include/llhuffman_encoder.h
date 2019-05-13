/*****************************************************************************
 *
 * LLHuff - Length-limited Huffman encoding
 *
 * released under LGPL-3.0
 *
 * 2017-2018 André Weißenberger
 *
 *****************************************************************************/

#ifndef LL_HUFFMAN_ENCODER_
#define LL_HUFFMAN_ENCODER_

#include "cuhd_constants.h"
#include "cuhd_codetable.h"
#include "llhuffman_encoder_table.h"

#include <memory>
#include <vector>

namespace llhuff {
    class LLHuffmanEncoder {

        public:
            struct Symbol {
                size_t length;
                size_t count;
                SYMBOL_TYPE symbol;
            };

            static std::shared_ptr<std::vector<Symbol>> get_symbol_lengths(
                SYMBOL_TYPE* data, size_t size);

            static std::shared_ptr<LLHuffmanEncoderTable> get_encoder_table(
                std::shared_ptr<std::vector<Symbol>> symbol_lengths);

            static void encode_memory(UNIT_TYPE* out, size_t size_out,
                SYMBOL_TYPE* in, size_t size_in,
                std::shared_ptr<LLHuffmanEncoderTable> encoder_table);

            static std::shared_ptr<cuhd::CUHDCodetable> get_decoder_table(
                std::shared_ptr<LLHuffmanEncoderTable> enc_table);

        private:
            struct Coin {
                float denomination;
                float numismatic;
                SYMBOL_TYPE symbol;
            };
            
            struct CoinPackage {
                float denomination;
                float numismatic;
                
                std::vector<Coin> coins;
            
                CoinPackage package(CoinPackage* p) {
                    CoinPackage np;
                    
                    np.denomination = denomination + p->denomination;
                    np.numismatic = numismatic + p->numismatic;
            
                    np.coins.insert(
                        std::end(np.coins), std::begin(coins), std::end(coins));
                    np.coins.insert(
                        std::end(np.coins), std::begin(p->coins),
                        std::end(p->coins));
                    
                    return np;
                }
            };
    };
}

#endif /* LL_HUFFMAN_ENCODER_H_ */

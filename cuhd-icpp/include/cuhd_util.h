/*****************************************************************************
 *
 * CUHD - A massively parallel Huffman decoder
 *
 * released under LGPL-3.0
 *
 * 2017-2018 André Weißenberger
 *
 *****************************************************************************/

#ifndef CUHD_UTIL_
#define CUHD_UTIL_

#include "cuhd_constants.h"

#include <string>
#include <functional>
#include <chrono>

namespace cuhd {
    class CUHDUtil {
        public:

            // calls given function and returns runtime in microseconds
            static std::pair<std::string, size_t> time(std::string s,
                std::function<void()> f);
            
            // easy to use timers
            #define TIMER_START(vec, label) vec.push_back(\
            cuhd::CUHDUtil::time(label, [&]() {
            
            #define TIMER_STOP }));
            
            // returns true if arrays equal, false otherwise        
            static bool equals(SYMBOL_TYPE* a, SYMBOL_TYPE* b, size_t size);

            // save integer division
            #define SDIV(n, m) ((n + m - 1) / m)
            
            // returns optimal subsequence size for given compression ratio
            static size_t optimal_subsequence_size(size_t input_size, 
                size_t output_size, size_t pref, size_t device_pref);
    };
}

#endif /* CUHD_UTIL_H_ */


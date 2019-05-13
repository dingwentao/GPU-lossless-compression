/*****************************************************************************
 *
 * CUHD - A massively parallel Huffman decoder
 *
 * released under LGPL-3.0
 *
 * 2017-2018 André Weißenberger
 *
 *****************************************************************************/

#include "cuhd_util.h"

std::pair<std::string, size_t> cuhd::CUHDUtil::time(std::string s,
    std::function<void()> f) {
	std::chrono::high_resolution_clock clock;
	std::chrono::nanoseconds duration;

	std::chrono::time_point<std::chrono::high_resolution_clock> start;
	std::chrono::time_point<std::chrono::high_resolution_clock> end;

    start = clock.now();

    f();
    
    end = clock.now(); 

    duration = end - start;

    std::pair<std::string, size_t> p(s,
        std::chrono::duration_cast<std::chrono::microseconds>
            (duration).count());

    return p;
}

bool cuhd::CUHDUtil::equals(SYMBOL_TYPE* a, SYMBOL_TYPE* b, size_t size) {
    for(size_t i = 0; i < size; ++i)
        if(a[i] != b[i]) return false;

    return true;
}

size_t cuhd::CUHDUtil::optimal_subsequence_size(size_t input_size, 
    size_t output_size, size_t pref, size_t device_pref) {
    
    float device = 1.0f / device_pref;
    size_t curr_size = pref;
    
    while(curr_size > 1 && (float) SDIV(SDIV(input_size, 4), curr_size) /
        (float) SDIV(output_size, 4) < device) {
        curr_size = SDIV(curr_size, 2);
    }
    
    return curr_size;
}


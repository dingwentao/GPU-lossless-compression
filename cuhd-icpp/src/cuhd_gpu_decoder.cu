/*****************************************************************************
 *
 * CUHD - A massively parallel Huffman decoder
 *
 * released under LGPL-3.0
 *
 * 2017-2018 André Weißenberger
 *
 *****************************************************************************/

#include "cuhd_gpu_decoder.h"

#include <thrust/device_ptr.h>
#include <thrust/scan.h>

__device__ __forceinline__ void decode_subsequence(
    std::uint32_t subsequence_size,
    std::uint32_t current_subsequence,
    UNIT_TYPE mask,
    std::uint32_t shift,
    std::uint32_t start_bit,
    std::uint32_t &in_pos,
    const UNIT_TYPE* __restrict__ in_ptr,
    UNIT_TYPE &window,
    UNIT_TYPE &next,
    std::uint32_t &last_word_unit,
    std::uint32_t &last_word_bit,
    std::uint32_t &num_symbols,
    std::uint32_t &out_pos,
    SYMBOL_TYPE* out_ptr,
    std::uint32_t &next_out_pos,
    const cuhd::CUHDCodetableItemSingle* __restrict__ table,
    const std::uint32_t bits_in_unit,
    std::uint32_t &last_at,
    bool overflow,
    bool write_output) {

    // current unit in this subsequence
    std::uint32_t current_unit = 0;
    
    // current bit position in unit
    std::uint32_t at = start_bit;

    // number of symbols found in this subsequence
    std::uint32_t num_symbols_l = 0;

    // perform overflow from previous subsequence
    if(overflow && current_subsequence > 0) {

        // shift to start
        UNIT_TYPE copy_next = next;
        copy_next >>= bits_in_unit - at;

        next <<= at;
        window <<= at;
        window += copy_next;

        // decode first symbol
        std::uint32_t taken = table[(window & mask) >> shift].num_bits;

        copy_next = next;
        copy_next >>= bits_in_unit - taken;

        next <<= taken;
        window <<= taken;
        at += taken;
        window += copy_next;

        // overflow
        if(at > bits_in_unit) {
            ++in_pos;
            window = in_ptr[in_pos];
            next = in_ptr[in_pos + 1];
            at -= bits_in_unit;
            window <<= at;
            next <<= at;

            copy_next = in_ptr[in_pos + 1];
            copy_next >>= bits_in_unit - at;
            window += copy_next;
        }

        else {
            ++in_pos;
            window = in_ptr[in_pos];
            next = in_ptr[in_pos + 1];
            at = 0;
        }
    }
    
    while(current_unit < subsequence_size) {
        
        while(at < bits_in_unit) {
            const cuhd::CUHDCodetableItemSingle hit =
                table[(window & mask) >> shift];
            
            // decode a symbol
            std::uint32_t taken = hit.num_bits;
            ++num_symbols_l;

            if(write_output) {
                if(out_pos < next_out_pos) {
                    out_ptr[out_pos] = hit.symbol;
                    ++out_pos;
                }
            }
            
            UNIT_TYPE copy_next = next;
            copy_next >>= bits_in_unit - taken;

            next <<= taken;
            window <<= taken;
            last_word_bit = at;
            at += taken;
            window += copy_next;
            last_word_unit = current_unit;
        }
        
        // refill decoder window if necessary
        ++current_unit;
        ++in_pos;
        
        window = in_ptr[in_pos];
        next = in_ptr[in_pos + 1];
        
        if(at == bits_in_unit) {
            at = 0;
        }

        else {
            at -= bits_in_unit;
            window <<= at;
            next <<= at;
            
            UNIT_TYPE copy_next = in_ptr[in_pos + 1];
            copy_next >>= bits_in_unit - at;
            window += copy_next;
        }
    }
    
    num_symbols = num_symbols_l;
    last_at = at;
}

__global__ void phase1_decode_subseq(
    std::uint32_t subsequence_size,
    std::uint32_t total_num_subsequences,
    std::uint32_t table_size,
    const UNIT_TYPE* __restrict__ in_ptr,
    const cuhd::CUHDCodetableItemSingle* __restrict__ table,
    uint4* sync_points,
    const std::uint32_t bits_in_unit) {
    
    const std::uint32_t gid = blockDim.x * blockIdx.x + threadIdx.x;
    
    if(gid < total_num_subsequences) {

        std::uint32_t current_subsequence = gid;
        std::uint32_t in_pos = gid * subsequence_size;

        // mask
        const UNIT_TYPE mask = ~(((UNIT_TYPE) (0) - 1) >> table_size);

        // shift right
        const std::uint32_t shift = bits_in_unit - table_size;

        std::uint32_t out_pos = 0;
        std::uint32_t next_out_pos = 0;
        std::uint8_t* out_ptr = 0;

        // sliding window
        UNIT_TYPE window = in_ptr[in_pos];
        UNIT_TYPE next = in_ptr[in_pos + 1];

        // start bit of last codeword in this subsequence
        std::uint32_t last_word_unit = 0;
        std::uint32_t last_word_bit = 0;

        // number of symbols found in this subsequence
        std::uint32_t num_symbols = 0;
        
        // bit position of next codeword
        std::uint32_t last_at = 0;
        
        std::uint32_t subsequences_processed = 0;
        bool synchronised_flag = false;   
        
        while(subsequences_processed < blockDim.x) {
        
            if(!synchronised_flag
                && current_subsequence < total_num_subsequences) {

                decode_subsequence(subsequence_size, current_subsequence,
                    mask, shift, last_at, in_pos, in_ptr, window, next,
                    last_word_unit, last_word_bit, num_symbols,
                    out_pos, out_ptr, next_out_pos, table, bits_in_unit,
                    last_at, false, false);
                
                if(subsequences_processed == 0) {
                    sync_points[current_subsequence] =
                        {last_word_unit, last_word_bit, num_symbols, 1};
                }

                else {
                    uint4 sync_point = sync_points[current_subsequence];
                    
                    // if sync point detected
                    if(sync_point.x == last_word_unit
                        && sync_point.y == last_word_bit) {
                            sync_point.z = num_symbols;
                            sync_point.w = 1;
                            synchronised_flag = true;
                    }
                    
                    // correct erroneous position data
                    else {
                        sync_point.x = last_word_unit;
                        sync_point.y = last_word_bit;
                        sync_point.z = num_symbols;
                        sync_point.w = 0;
                    }
                    
                    sync_points[current_subsequence] = sync_point;
                }
            }
            
            ++current_subsequence;
            ++subsequences_processed;
            
            __syncthreads();
        }
    }
}

__global__ void phase2_synchronise_blocks(
    std::uint32_t subsequence_size,
    std::uint32_t total_num_subsequences,
    std::uint32_t table_size,
    std::uint32_t num_blocks,
    const UNIT_TYPE* __restrict__ in_ptr,
    const cuhd::CUHDCodetableItemSingle* __restrict__ table,
    uint4* sync_points,
    SYMBOL_TYPE* block_synchronised,
    const std::uint32_t bits_in_unit) {
    
    const std::uint32_t gid = blockDim.x * blockIdx.x + threadIdx.x;
    const std::uint32_t num_of_seams = num_blocks - 1;

    if(gid < num_of_seams) {
    
        // mask
        const UNIT_TYPE mask = ~(((UNIT_TYPE) (0) - 1) >> table_size);

        // shift
        const std::uint32_t shift = bits_in_unit - table_size;

        std::uint32_t out_pos = 0;
        std::uint32_t next_out_pos = 0;
        std::uint8_t* out_ptr = 0;
        
        // jump to first sequence of the block
        std::uint32_t current_subsequence = (gid + 1) * blockDim.x;
        
        // search for synchronised sequences at the end of previous block
        uint4 sync_point = sync_points[current_subsequence - 1];
        
        // current unit
        std::uint32_t in_pos = (current_subsequence - 1) * subsequence_size;
        
        // start bit of last codeword in this subsequence
        std::uint32_t last_word_unit = 0;
        std::uint32_t last_word_bit = 0;

        // number of symbols found in this subsequence
        std::uint32_t num_symbols = 0;
        
        std::uint32_t last_at = sync_point.y;
        
        in_pos += sync_point.x;
        
        // sliding window
        UNIT_TYPE window = in_ptr[in_pos];
        UNIT_TYPE next = in_ptr[in_pos + 1];
        
        std::uint32_t subsequences_processed = 0;
        bool synchronised_flag = false;

        while(subsequences_processed < blockDim.x) {
        
            if(!synchronised_flag) {
                decode_subsequence(subsequence_size, current_subsequence,
                    mask, shift, last_at, in_pos, in_ptr,
                    window, next, last_word_unit, last_word_bit, num_symbols,
                    out_pos, out_ptr, next_out_pos, table, bits_in_unit,
                    last_at, true, false);
                
                sync_point = sync_points[current_subsequence];
                
                // if sync point detected
                if(sync_point.x == last_word_unit
                    && sync_point.y == last_word_bit) {
                        sync_point.z = num_symbols;
                        sync_point.w = 1;
                    
                        block_synchronised[gid + 1] = 1;
                        synchronised_flag = true;
                }
                
                // correct erroneous position data
                else {
                    sync_point.x = last_word_unit;
                    sync_point.y = last_word_bit;
                    sync_point.z = num_symbols;
                    sync_point.w = 0;
                    block_synchronised[gid + 1] = 0;
                }
                
                sync_points[current_subsequence] = sync_point;
            }
            
            ++current_subsequence;
            ++subsequences_processed;
            
            __syncthreads();
        }
    }
}

__global__ void phase3_copy_num_symbols_from_sync_points_to_aux(
    std::uint32_t total_num_subsequences,
    const uint4* __restrict__ sync_points,
    std::uint32_t* subsequence_output_sizes) {

    const std::uint32_t gid = blockDim.x * blockIdx.x + threadIdx.x;

    if(gid < total_num_subsequences) {
        subsequence_output_sizes[gid] = sync_points[gid].z;
    }
}

__global__ void phase3_copy_num_symbols_from_aux_to_sync_points(
    std::uint32_t total_num_subsequences,
    uint4* sync_points,
    const std::uint32_t* __restrict__ subsequence_output_sizes) {

    const std::uint32_t gid = blockDim.x * blockIdx.x + threadIdx.x;

    if(gid < total_num_subsequences) {
        sync_points[gid].z = subsequence_output_sizes[gid];
    }
}

__global__ void phase4_decode_write_output(
    std::uint32_t subsequence_size,
    std::uint32_t total_num_subsequences,
    std::uint32_t table_size,
    const UNIT_TYPE* __restrict__ in_ptr,
    SYMBOL_TYPE* out_ptr,
    std::uint32_t output_size,
    cuhd::CUHDCodetableItemSingle* table,
    const uint4* __restrict__ sync_points,
    const std::uint32_t bits_in_unit) {
    
    const std::uint32_t gid = blockDim.x * blockIdx.x + threadIdx.x;

    if(gid < total_num_subsequences) {
        
        // mask
        const UNIT_TYPE mask = ~(((UNIT_TYPE) (0) - 1) >> table_size);

        // shift
        const size_t shift = bits_in_unit - table_size;

        // start bit of last codeword in this subsequence
        std::uint32_t last_word_unit = 0;
        std::uint32_t last_word_bit = 0;
        
        // number of symbols found in this subsequence
        std::uint32_t num_symbols = 0;
        
        // bit position of next codeword
        std::uint32_t last_at = 0;
        
        std::uint32_t current_subsequence = gid;
        std::uint32_t in_pos = current_subsequence * subsequence_size;
        
        uint4 sync_point = sync_points[current_subsequence];
        uint4 next_sync_point = sync_points[current_subsequence + 1];
        
        std::uint32_t out_pos = sync_point.z;
        std::uint32_t next_out_pos = gid == total_num_subsequences - 1 ?
            output_size : next_sync_point.z;
        
        if(gid > 0) {
            sync_point = sync_points[current_subsequence - 1];
            in_pos = (current_subsequence - 1) * subsequence_size;
        }
        
        // sliding window
        UNIT_TYPE window = in_ptr[in_pos];
        UNIT_TYPE next = in_ptr[in_pos + 1];
        
        // start bit
        std::uint32_t start = 0;
        
        if(gid > 0) {
            in_pos += sync_point.x;
            start = sync_point.y;

            window = in_ptr[in_pos];
            next = in_ptr[in_pos + 1];
        }

        // overflow from previous subsequence, decode, write output
        decode_subsequence(subsequence_size, current_subsequence, mask, shift,
            start, in_pos, in_ptr, window, next,
            last_word_unit, last_word_bit, num_symbols, out_pos, out_ptr,
            next_out_pos, table, bits_in_unit, last_at, true, true);
    }
}

void cuhd::CUHDGPUDecoder::decode(
    std::shared_ptr<cuhd::CUHDGPUInputBuffer> input,
    size_t input_size,
    std::shared_ptr<cuhd::CUHDGPUOutputBuffer> output,
    size_t output_size,
    std::shared_ptr<cuhd::CUHDGPUCodetable> table,
    std::shared_ptr<cuhd::CUHDGPUDecoderMemory> aux,
    size_t max_codeword_length,
    size_t preferred_subsequence_size,
    size_t threads_per_block) {
    
    UNIT_TYPE* in_ptr = input->get();
    SYMBOL_TYPE* out_ptr = output->get();
    cuhd::CUHDCodetableItemSingle* table_ptr = table->get();
    
    uint4* sync_info = reinterpret_cast<uint4*>(aux->get_sync_info());
    std::uint32_t* output_sizes = aux->get_output_sizes();
    std::uint8_t* sequence_synced_device = aux->get_sequence_synced_device();
    std::uint8_t* sequence_synced_host = aux->get_sequence_synced_host();

    size_t num_subseq = SDIV(input_size, preferred_subsequence_size);
    size_t num_sequences = SDIV(num_subseq, threads_per_block);
    
    const std::uint32_t bits_in_unit = sizeof(UNIT_TYPE) * 8;

    // launch phase 1 (intra-sequence synchronisation)
    phase1_decode_subseq<<<num_sequences, threads_per_block>>>(
        preferred_subsequence_size,
        num_subseq,
        max_codeword_length,
        in_ptr,
        table_ptr,
        sync_info,
        bits_in_unit);
    CUERR
    
    // launch phase 2 (inter-sequence synchronisation)
    bool blocks_synchronised = true;
    
    do {
        phase2_synchronise_blocks<<<num_sequences, threads_per_block>>>(
                preferred_subsequence_size,
                num_subseq,
                max_codeword_length,
                num_sequences,
                in_ptr,
                table_ptr,
                sync_info,
                sequence_synced_device,
                bits_in_unit);
        CUERR

        aux->retrieve_sync_data();
        
        const size_t num_blocks = num_sequences - 1;
        
        bool zero_found = false;
        
        for(size_t i = 1; i < num_blocks; ++i) {
            if(sequence_synced_host[i] == 0) {
                zero_found = true;
                break;
            }
        }
        
        if(zero_found) {
            blocks_synchronised = false;
        }
        
        else {
            blocks_synchronised = true;
        }
        
    } while(!blocks_synchronised);
    
    // launch phase 3 (parallel prefix sum)
    thrust::device_ptr<std::uint32_t> thrust_sync_points(output_sizes);

    phase3_copy_num_symbols_from_sync_points_to_aux<<<
        num_sequences, threads_per_block>>>(num_subseq, sync_info, output_sizes);
    CUERR

    thrust::exclusive_scan(thrust_sync_points,
        thrust_sync_points + num_subseq, thrust_sync_points);

    phase3_copy_num_symbols_from_aux_to_sync_points<<<
        num_sequences, threads_per_block>>>(num_subseq, sync_info, output_sizes);
    CUERR
    
    // launch phase 4 (final decoding)
    phase4_decode_write_output<<<num_sequences, threads_per_block>>>(
            preferred_subsequence_size,
            num_subseq,
            max_codeword_length,
            in_ptr,
            out_ptr,
            output_size,
            table_ptr,
            sync_info,
            bits_in_unit);
    CUERR
}


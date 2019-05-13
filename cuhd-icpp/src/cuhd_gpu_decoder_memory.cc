/*****************************************************************************
 *
 * CUHD - A massively parallel Huffman decoder
 *
 * released under LGPL-3.0
 *
 * 2017-2018 André Weißenberger
 *
 *****************************************************************************/

#include "cuhd_gpu_decoder_memory.h"

cuhd::CUHDGPUDecoderMemory::CUHDGPUDecoderMemory(
    size_t num_units, size_t subsequence_size, size_t num_threads)
    : num_subseq_(SDIV(num_units, subsequence_size)),
      num_blocks_(SDIV(num_subseq_, num_threads)),
      is_allocated_(false) {
      
}

cuhd::CUHDGPUDecoderMemory::~CUHDGPUDecoderMemory() {
    free();
}

void cuhd::CUHDGPUDecoderMemory::allocate() {
    if(is_allocated_) return;

    // synchronisation info for each subsequence
    cudaMalloc((void**) &sync_info_,
        num_subseq_ * sizeof(CUHDSubsequenceSyncPoint));
    CUERR

    // size of output for each subsequence
    cudaMalloc((void**) &output_sizes_,
        num_subseq_ * sizeof(std::uint32_t));
    CUERR

    // flags for indicating inter-sequence synchronisation
    cudaMalloc((void**) &sequence_synced_device_,
        num_blocks_ * sizeof(std::uint8_t));
    CUERR

    cudaMallocHost((void**) &sequence_synced_host_,
        num_blocks_ * sizeof(std::uint8_t));
    CUERR

    is_allocated_ = true;
}

void cuhd::CUHDGPUDecoderMemory::retrieve_sync_data() {

    // copy sync data to host
    cudaMemcpy(sequence_synced_host_, sequence_synced_device_,
        num_blocks_ * sizeof(std::uint8_t), cudaMemcpyDeviceToHost);
    CUERR
}

void cuhd::CUHDGPUDecoderMemory::free() {
    if(!is_allocated_) return;
    
    cudaFree(sync_info_);
	CUERR

	cudaFree(output_sizes_);
    CUERR

    cudaFree(sequence_synced_device_);
	CUERR

	cudaFreeHost(sequence_synced_host_);
    CUERR
    
    is_allocated_ = false;
}

cuhd::CUHDSubsequenceSyncPoint* cuhd::CUHDGPUDecoderMemory::get_sync_info() {
    return sync_info_;
}

std::uint32_t* cuhd::CUHDGPUDecoderMemory::get_output_sizes() {
    return output_sizes_;
}

std::uint8_t* cuhd::CUHDGPUDecoderMemory::get_sequence_synced_device() {
    return sequence_synced_device_;
}

std::uint8_t* cuhd::CUHDGPUDecoderMemory::get_sequence_synced_host() {
    return sequence_synced_host_;   
}


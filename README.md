GPU CUDA Lossless Compressors Review

GPU-accelerated Lossless Compression
1. BZIP/BWT Family

CUDPP - CUDA Data-Parallel Primitives Library
Abstract: CUDPP release 2.1 is a feature release that adds many new parallel primitives. We have added cudppCompress, a lossless data compression algorithm. This compression utilizes efficient parallel Burrows-Wheeler and Move-to-Front transforms (BTW) which are also exposed through the API. GPU implementation is dominated by BWT performance and is 2.78× slower than bzip2, with BWT and MTF-Huffman respectively 2.89x and 1.34x slower.
Code: http://cudpp.github.io/
Paper: [InPar’12] Patel, R.A., Zhang, Y., Mak, J., Davidson, A. and Owens, J.D., 2012. Parallel lossless data compression on the GPU (pp. 1-9). IEEE.
Slides: http://on-demand.gputechconf.com/gtc/2012/presentations/S0361-Lossless-Data-Compression-on-GPUs.pdf
Comment: Cuda 10 is not working, Cuda 9.1 seems working according to https://github.com/cudpp/cudpp/issues/185.

LibBSC - High Performance Block-Sorting Data Compression Library
Abstract: A library for lossless, block-sorting data compression. bsc is a high-performance file compressor based on lossless, block-sorting data compression algorithms. libbsc is a library based on bsc, it uses the same algorithms as bsc and enables you to compress memory blocks.
Code: https://github.com/IlyaGrebnov/libbsc, http://libbsc.com/
Paper: Burrows, M. and Wheeler, D.J., 1994. A block-sorting lossless data compression algorithm. (https://www.hpl.hp.com/techreports/Compaq-DEC/SRC-RR-124.pdf)

Parallel Variable-Length Encoding on GPGPUs
Abstract: bzip2-cuda is a parallel version of the BZip2 lossless data compression algorithm using NVIDIA's CUDA, with the aim of improving compression/decompression speed, leaving the compression ratio intact. bzip2 compression program is based on Burrows–Wheeler algorithm.
Code: https://github.com/bzip2-cuda/bzip2-cuda, http://bzip2-cuda.github.io/

Fast Burrows Wheeler Compression Using All-Cores (Heterogeneous)
Abstract: We present an all-core implementation of Burrows Wheeler Compression algorithm that exploits all computing resources on a system. We develop a fast GPU BWC algorithm by extending the state-of-the-art GPU string sort to efficiently perform BWT step of BWC. Our hybrid BWC with GPU acceleration achieves a 2.9× speedup over best CPU implementation.
Code: https://github.com/aditya12agd5/cuda_bzip2
Paper: [IPDPSW’15] Deshpande, A. and Narayanan, P.J., 2015, May. Fast Burrows Wheeler Compression Using All-Cores. In 2015 IEEE International Parallel and Distributed Processing Symposium Workshop (pp. 628-636). IEEE.

[compress]
./bzip2 -kf -9 -n0 ~/Data/Hurricane/Uf01.bin.f32
[DEBUG] numThreads = 0
Block size : 900000
numThreads read 0
Number of additional CPU threads 0
[Time thread1] 11.695528
[Time thread2 work] 9.547885
[Time thread2] 12.159963
Number of CPU threads 2
Out of the total 111 blocks GPU did 111
total compression time (with overlap) 12.160248

[DEBUG] numThreads = 1
Block size : 900000
numThreads read 1
Number of additional CPU threads 1
[Time thread1] 5.819389
[Time thread2 work] 4.930835
[Time thread2] 6.711509
Number of CPU threads 3
Out of the total 111 blocks GPU did 57
total compression time (with overlap) 6.713813

[decompress]
./bzip2 -d ~/Data/Hurricane/Uf01.bin.f32.bz2
bzip2: /home/dtao/Data/Hurricane/Uf01.bin.f32.bz2: trailing garbage after EOF ignored
Total compression time 8.180155

Note: The Burrows–Wheeler transform (BWT, also called block-sorting compression) rearranges a character string into runs of similar characters. This is useful for compression, since it tends to be easy to compress a string that has runs of repeated characters by techniques such as move-to-front transform and run-length encoding. More importantly, the transformation is reversible, without needing to store any additional data except the position of the first original character. The BWT is thus a "free" method of improving the efficiency of text compression algorithms, costing only some extra computation. bzip2 is a free and open-source file compression program that uses the Burrows–Wheeler algorithm.

2. Huffman Family

E2MC - Entropy Encoding Based Memory Compression for GPUs
Abstract: We propose an entropy encoding based memory compression (E2MC) technique for GPUs, which is based on the well-known Huffman encoding. We study the feasibility of entropy encoding for GPUs and show that it achieves higher compression ratios than state-of-the-art GPU compression techniques. Furthermore, we address the key challenges of probability estimation, choosing an appropriate symbol length for encoding, and decompression with low latency.
Code: ???
Paper: [IPDPS’17] Lal, S., Lucas, J. and Juurlink, B., 2017, May. E^ 2MC: Entropy Encoding Based Memory Compression for GPUs. In 2017 IEEE International Parallel and Distributed Processing Symposium (IPDPS) (pp. 1119-1128). IEEE.

CUHD - A Massively Parallel Huffman Decoder 
Abstract: A Huffman decoder for processing raw (i.e. unpartitioned) Huffman encoded data on the GPU. It also includes a basic, sequential encoder.
Code: https://github.com/weissenberger/gpuhd
Paper: [ICPP’16] Sitaridi, E., Mueller, R., Kaldewey, T., Lohman, G. and Ross, K.A., 2016, August. Massively-parallel lossless data decompression. In 2016 45th International Conference on Parallel Processing (ICPP) (pp. 242-247). IEEE.
Slides: http://on-demand.gputechconf.com/gtc/2014/presentations/S4459-parallel-lossless-compression-using-gpus.pdf

Unknown CUDA Huffman code: https://github.com/smadhiv/HuffmanCoding_MPI_CUDA 

3. LZ Family

CULZSS: LZSS Lossless Data Compression on CUDA
Abstract: We present an implementation of Lempel-Ziv-Storer-Szymanski (LZSS) lossless data compression algorithm by using NVIDIA GPUs CUDA Framework. Our system outperforms the serial CPU LZSS implementation by up to 18x, the parallel threaded version up to 3x and the BZIP2 program by up to 6x in terms of compression time.
Code: https://github.com/adnanozsoy/CUDA_Compression
Paper: [CLUSTER’11] Ozsoy, A. and Swany, M., 2011, September. CULZSS: LZSS lossless data compression on CUDA. In 2011 IEEE International Conference on Cluster Computing (pp. 403-411). IEEE.

Run-length encoding in CUDA 
Code: https://github.com/mgrzeszczak/cuda-run-length-encoding
Paper: https://erkaman.github.io/posts/cuda_rle.html

Unknown CUDA LZSS code: https://github.com/abshkbh/cuda-lzss

# GPU-accelerated Lossless Compression Survey

https://github.com/dingwentao/GPU-lossless-compression

## **BZIP/BWT Family**

### 1. CUDPP - CUDA Data-Parallel Primitives Library

Abstract: CUDPP release 2.1 is a feature release that adds many new parallel primitives. We have added cudppCompress, a lossless data compression algorithm. This compression utilizes efficient parallel Burrows-Wheeler and Move-to-Front transforms (BTW) which are also exposed through the API. GPU implementation is dominated by BWT performance and is 2.78× slower than bzip2, with BWT and MTF-Huffman respectively 2.89x and 1.34x slower.

Code: http://cudpp.github.io/

Paper: [InPar’12] Patel, R.A., Zhang, Y., Mak, J., Davidson, A. and Owens, J.D., 2012. Parallel lossless data compression on the GPU (pp. 1-9). IEEE.
Slides: http://on-demand.gputechconf.com/gtc/2012/presentations/S0361-Lossless-Data-Compression-on-GPUs.pdf

*Cuda 10 is not working, Cuda 9.1 seems working according to https://github.com/cudpp/cudpp/issues/185.*

### 2. LibBSC - High Performance Block-Sorting Data Compression Library

Abstract: A library for lossless, block-sorting data compression. bsc is a high-performance file compressor based on lossless, block-sorting data compression algorithms. libbsc is a library based on bsc, it uses the same algorithms as bsc and enables you to compress memory blocks.

Code: https://github.com/IlyaGrebnov/libbsc, http://libbsc.com/

Paper: Burrows, M. and Wheeler, D.J., 1994. A block-sorting lossless data compression algorithm. (https://www.hpl.hp.com/techreports/Compaq-DEC/SRC-RR-124.pdf)

    [compress]
    ./bsc e /home/dtao/GPU-lossless-compression/testdata/largefile /home/dtao/GPU-lossless-compression/testdata/largefile.bsc -G
    This is bsc, Block Sorting Compressor. Version 3.1.0. 8 July 2012.
    Copyright (c) 2009-2012 Ilya Grebnov <Ilya.Grebnov@gmail.com>.

    /home/dtao/GPU-lossless-compression/testdata/largefile compressed 3569598 into 159230 in 0.147 seconds.

    [decompress]
    ./bsc d /home/dtao/GPU-lossless-compression/testdata/largefile.bsc /home/dtao/GPU-lossless-compression/testdata/largefile.bsc.out -G
    This is bsc, Block Sorting Compressor. Version 3.1.0. 8 July 2012.
    Copyright (c) 2009-2012 Ilya Grebnov <Ilya.Grebnov@gmail.com>.

    /home/dtao/GPU-lossless-compression/testdata/largefile. decompressed 159230 into 3569598 in 0.215 seconds.

### 3. Parallel Variable-Length Encoding on GPGPUs

Abstract: bzip2-cuda is a parallel version of the BZip2 lossless data compression algorithm using NVIDIA's CUDA, with the aim of improving compression/decompression speed, leaving the compression ratio intact. bzip2 compression program is based on Burrows–Wheeler algorithm.

Code: https://github.com/bzip2-cuda/bzip2-cuda, http://bzip2-cuda.github.io/

### 4. Fast Burrows Wheeler Compression Using All-Cores (Heterogeneous)

Abstract: We present an all-core implementation of Burrows Wheeler Compression algorithm that exploits all computing resources on a system. We develop a fast GPU BWC algorithm by extending the state-of-the-art GPU string sort to efficiently perform BWT step of BWC. Our hybrid BWC with GPU acceleration achieves a 2.9× speedup over best CPU implementation.

Code: https://github.com/aditya12agd5/cuda_bzip2

Paper: [IPDPSW’15] Deshpande, A. and Narayanan, P.J., 2015, May. Fast Burrows Wheeler Compression Using All-Cores. In 2015 IEEE International Parallel and Distributed Processing Symposium Workshop (pp. 628-636). IEEE.

    [compress]
    ./bzip2 -kf -9 -n0 /home/dtao/GPU-lossless-compression/testdata/largefile
    [DEBUG] numThreads = 0
    Block size : 900000
    numThreads read 0
    Number of additional CPU threads 0
    [Time thread1] 5.133659
    [Time thread2 work] 0.135336
    [Time thread2] 5.165935
    Number of CPU threads 2
    Out of the total 4 blocks GPU did 4
    total compression time (with overlap) 5.166217

    ./bzip2 -kf -9 -n1 /home/dtao/GPU-lossless-compression/testdata/largefile
    [DEBUG] numThreads = 1
    Block size : 900000
    numThreads read 1
    Number of additional CPU threads 1
    [Time thread1] 2.147425
    [Time thread2 work] 0.036489
    [Time thread2] 2.183931
    Number of CPU threads 3
    Out of the total 4 blocks GPU did 1
    total compression time (with overlap) 2.185086

    [decompress]
    ./bzip2 -d  /home/dtao/GPU-lossless-compression/testdata/largefile.bz2
    Total compression time 0.190715

    Compression ratio = 4.4

Note: The Burrows–Wheeler transform (BWT, also called block-sorting compression) rearranges a character string into runs of similar characters. This is useful for compression, since it tends to be easy to compress a string that has runs of repeated characters by techniques such as move-to-front transform and run-length encoding. More importantly, the transformation is reversible, without needing to store any additional data except the position of the first original character. The BWT is thus a "free" method of improving the efficiency of text compression algorithms, costing only some extra computation. bzip2 is a free and open-source file compression program that uses the Burrows–Wheeler algorithm.

## **Huffman Family**

### 5. E2MC - Entropy Encoding Based Memory Compression for GPUs

Abstract: We propose an entropy encoding based memory compression (E2MC) technique for GPUs, which is based on the well-known Huffman encoding. We study the feasibility of entropy encoding for GPUs and show that it achieves higher compression ratios than state-of-the-art GPU compression techniques. Furthermore, we address the key challenges of probability estimation, choosing an appropriate symbol length for encoding, and decompression with low latency.

Code: ???

Paper: [IPDPS’17] Lal, S., Lucas, J. and Juurlink, B., 2017, May. E^ 2MC: Entropy Encoding Based Memory Compression for GPUs. In 2017 IEEE International Parallel and Distributed Processing Symposium (IPDPS) (pp. 1119-1128). IEEE.

### 6. CUHD - A Massively Parallel Huffman Decoder 

Abstract: A Huffman decoder for processing raw (i.e. unpartitioned) Huffman encoded data on the GPU. It also includes a basic, sequential encoder.

Code: https://github.com/weissenberger/gpuhd

Paper: [ICPP’16] Sitaridi, E., Mueller, R., Kaldewey, T., Lohman, G. and Ross, K.A., 2016, August. Massively-parallel lossless data decompression. In 2016 45th International Conference on Parallel Processing (ICPP) (pp. 242-247). IEEE.
Slides: http://on-demand.gputechconf.com/gtc/2014/presentations/S4459-parallel-lossless-compression-using-gpus.pdf

    [compress/decompress]
    ./demo 0 100
    generating random data.. 64527550µs
    generating coding tables.. 1003450µs
    encoding.. 1274812µs
    GPU buffer allocation.. 509326µs
    GPU/Host aux memory allocation.. 2601µs
    GPU memcpy HtD.. 14547µs
    decoding.. 1520µs
    GPU memcpy DtH.. 24461µs

### 7. Unknown CUDA Huffman

Code: https://github.com/smadhiv/HuffmanCoding_MPI_CUDA 

## **LZ Family**

### 8. CULZSS: LZSS Lossless Data Compression on CUDA

Abstract: We present an implementation of Lempel-Ziv-Storer-Szymanski (LZSS) lossless data compression algorithm by using NVIDIA GPUs CUDA Framework. Our system outperforms the serial CPU LZSS implementation by up to 18x, the parallel threaded version up to 3x and the BZIP2 program by up to 6x in terms of compression time.

Code: https://github.com/adnanozsoy/CUDA_Compression

Paper: [CLUSTER’11] Ozsoy, A. and Swany, M., 2011, September. CULZSS: LZSS lossless data compression on CUDA. In 2011 IEEE International Conference on Cluster Computing (pp. 403-411). IEEE.

    [compress]
    ./main -i  /home/dtao/GPU-lossless-compression/testdata/largefile -o /home/dtao/GPU-lossless-compression/testdata/largefile.lzss
    file size:3569598 eachblock_size:1048576 num_of_blocks:4 padding:624706
    Initializing the GPU
    All the time took:  1.013202
    Throughput for 4 runs of 3MB is :   23.687281Mbps

    [decompress]
    ./main -d 1 -i /home/dtao/GPU-lossless-compression/testdata/largefile.lzss -o /home/dtao/GPU-lossless-compression/testdata/largefile.lzss.out
    Doing decompression
    Num bufs 4 padding size 624706 bufsize 1048576
    Initializing the GPU
    All the time took:  0.802943
    Throughput for  2MB is :    19.926695Mbps

    Compression Ratio = 1.60

### 9. Unknown CUDA LZSS

Code: https://github.com/abshkbh/cuda-lzss

    [GPU compress]
    ./cuda_main -f /home/dtao/GPU-lossless-compression/testdata/largefile
    Before encoding
    Numblocks = 72
    Time for CPU to GPU is 0sec and 794usec
    Time for GPU to CPU is 0sec and 2254usec
    Total compressed output : 3092074
    Compression Ratio : 1.154435
    Only Kernel Time is 1sec and 122868usec
    Total Time is 14sec and 875701usec
    After encoding

    [CPU compress]
    ./sample -c -i /home/dtao/GPU-lossless-compression/testdata/largefile -o /home/dtao/GPU-lossless-compression/testdata/largefile.lzss2
    Before encoding
    T1 is 1557765394sec and 124395usec
    T2 is 1557765400sec and 706787usec
    Compression Ratio: 1.925268
     


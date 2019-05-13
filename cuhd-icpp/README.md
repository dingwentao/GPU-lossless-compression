# CUHD - A Massively Parallel Huffman Decoder

A Huffman decoder for processing raw (i.e. unpartitioned) Huffman encoded data on the GPU. It also includes a basic, sequential encoder.

For further information, please refer to our [conference paper](https://doi.org/10.1145/3225058.3225076).

## Requirements

* CUDA-enabled GPU with compute capability 3.0 or higher
* GNU/Linux
* GNU compiler version 5.4.0 or higher
* CUDA SDK 8 or higher
* latest proprietary graphics drivers

## Compilation process

### Configuration

Please edit the Makefile:

1. Set `CUDA_INCLUDE` to the include directory of your CUDA installation, e.g.: `CUDA_INCLUDE = /usr/local/cuda-9.1/include`

2. Set `CUDA_LIB` to the library directory of your CUDA installation, e.g.: `CUDA_LIB = /usr/local/cuda-9.1/lib64`

3. Set `ARCH` to the compute capability of your GPU, i.e. `ARCH = 35` for compute capability 3.5. If you'd like to compile the decoder for multiple generations of GPUs, please edit `NVCC_FLAGS` accordingly.

### Test program

The test program will generate a chunk of random, binomially distributed data, encode the data with a specified maximum codeword length and decode the data on the GPU.

#### Compiling the test program

To compile the test program, configure the Makefile as described above. Run:

`make`

#### Running the test program

`./bin/demo <compute device index> <size of input in megabytes>`

### Compiling a static library

To compile a static library, run:

`make lib`

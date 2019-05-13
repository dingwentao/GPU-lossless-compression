
#define BZ_GPU
#include "bzlib_private.h"
#include <stdio.h>
#include <algorithm>
#include <fstream>
#include <iterator>
#include <iostream>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/scan.h>
#include <thrust/scatter.h>
#include <thrust/iterator/zip_iterator.h>
#include <cuda.h>
#include <time.h>

#define MAX_THREADS_PER_BLOCK 1024

//#define __DEBUG__

__device__ unsigned int *global_int_original_string = NULL; 
__device__ unsigned int *global_first_sort_rank = NULL;

class Bar
{
	unsigned int *functor_string;
	unsigned int *functor_first_sort_rank;
	const int currentOffset;
	const int currentLength;  
	const int totalLength; 

	public:
		__host__
		Bar(int _currentOffset, int _currentLength, int _totalLength):functor_string(global_int_original_string), functor_first_sort_rank(global_first_sort_rank), currentOffset(_currentOffset), currentLength(_currentLength), totalLength(_totalLength) { }
	
		inline __device__
		bool operator() (thrust::tuple< unsigned int, unsigned int > t1, thrust::tuple< unsigned int, unsigned int > t2) { 
			int seg1 = thrust::get<0>(t1); 
			int seg2 = thrust::get<0>(t2);
			
			if(seg1 > seg2) return false;
			if(seg1 < seg2) return true;
			
			int ind1 = thrust::get<1>(t1);
			int ind2 = thrust::get<1>(t2);
			int count = 0;
			
			while( count < currentLength) {
				int newInd1 = ( ind1 + currentOffset + count ) %totalLength;
				int newInd2 = ( ind2 + currentOffset + count ) %totalLength;
				unsigned int a1 = (functor_string[newInd1]); 
				unsigned int a2 = (functor_string[newInd2]);
				int a = a1 - a2; 
				count+=4;
				if( a == 0 ) { continue; }
				else {
					if((a1>>24) > (a2>>24)) { return false;}
					else if((a1>>24) < (a2>>24)) {return true;} 
					else if((a1>>16) > (a2>>16)) { return false;}
					else if((a1>>16) < (a2>>16)) {return true;} 
					else if((a1>>8) > (a2>>8)) { return false;}
					else if((a1>>8) < (a2>>8)) {return true;} 
					else if((a1) > (a2)) { return false;}
					else if((a1) < (a2)) {return true;} 
				}
			}
			return false;
		}
};

__global__ void pack4CharsToInt(unsigned char *input_string, unsigned int *static_input_string, unsigned int *output_string, unsigned int *output_index, unsigned int *d_array_first_sort_rank, int length) { 
	
	int threadID = (blockIdx.x * blockDim.x) +  threadIdx.x;

	if(threadID > length) return;

	d_array_first_sort_rank[threadID] = 0;
	if(threadID % 3 != 0) { 
		int mult3 =(int) ((1.0*threadID)/3);
	
		int newIndex = mult3*2 + ((threadID % 3) - 1);

		output_index[newIndex] = threadID;
	
		output_string[newIndex] = (((unsigned int)input_string[threadID]) << 24) + 
					  (((unsigned int)input_string[(threadID+1) % length]) << 16) +  
	                                  (((unsigned int)input_string[(threadID+2) % length]) << 8) + 
					  (((unsigned int)input_string[(threadID+3) % length]));
	}

	static_input_string[threadID] =  (((unsigned int)input_string[threadID]) << 24) + 
					 (((unsigned int)input_string[(threadID+1) % length]) << 16) +  
	                                 (((unsigned int)input_string[(threadID+2) % length]) << 8) + 
					 (((unsigned int)input_string[(threadID+3) % length]));

}

__global__ void findSuccessor( unsigned int *d_array_original_string, unsigned int *d_array_string, unsigned int *d_array_index, unsigned int *d_array_segment, unsigned int *d_array_string_out, unsigned int *d_array_segment_out, int length, int originalLength, int sequenceCount) {

	int threadID = (blockIdx.x * blockDim.x) +  threadIdx.x;
	if(threadID > length) return;
	d_array_segment_out[threadID] = 0; 
	if(threadID > 0) { 
		if(((d_array_string[threadID]!=d_array_string[threadID-1]) || (d_array_segment[threadID]!=d_array_segment[threadID-1])) ) { 
			d_array_segment_out[threadID] = 1; 
		}
	}
	int successorIndex = (d_array_index[threadID] + sequenceCount + 4)%originalLength; 
	d_array_string_out[threadID] = d_array_original_string[successorIndex];
}


__global__ void  eliminateSizeOneKernel1(unsigned int *d_array_original_string, unsigned int *d_array_final_index, unsigned int *d_array_index, unsigned int *d_array_static_index, unsigned int *d_array_map, unsigned int *d_array_stencil, unsigned int *d_array_first_sort_rank, int sequenceCount, int length, int originalLength) {

	int threadID = (blockIdx.x * blockDim.x) +  threadIdx.x;

	if(threadID >= length) return;

	d_array_stencil[threadID] = 1;

        if(threadID == 0 && (d_array_map[threadID + 1] == 1)) { 
		d_array_stencil[threadID] = 0; 
	} else if( (threadID == (length-1)) && (d_array_map[threadID] == 1) ) {
		d_array_stencil[threadID] = 0;  
	} else if( (d_array_map[threadID] == 1) && (d_array_map[threadID + 1] == 1)) { 
		d_array_stencil[threadID] = 0; 
	}

	if(d_array_stencil[threadID] == 0) {
		int finalIndex = d_array_index[threadID];
		d_array_final_index[ d_array_static_index[threadID] ] = finalIndex;
		d_array_first_sort_rank[finalIndex] = d_array_static_index[threadID];
	}
}

__global__ void updateSegments( unsigned int *d_int_array_string, unsigned int *d_array_index, unsigned int *d_array_segment, unsigned int *d_array_segment_out, int size, int offset, int length, int originalLength) {

	int threadID = (blockIdx.x * blockDim.x) +  threadIdx.x;
	if(threadID > size) return;
	d_array_segment_out[threadID] = 0; 
	if(threadID > 0) { 
		if((d_array_segment[threadID - 1] != d_array_segment[threadID])) { 
			d_array_segment_out[threadID] = 1;
			return; 
		}
		int count = 0;
		unsigned int ind1 = d_array_index[threadID - 1]; 
		unsigned int ind2 = d_array_index[threadID]; 
		while(count < length) {
			if( d_int_array_string[(ind1 + offset + count) % originalLength] != d_int_array_string[(ind2 + offset + count) % originalLength]) { 
				d_array_segment_out[threadID] = 1; 
				break;
			}
			count+=4; 
		} 
	}
	return;
}


__global__ void  eliminateSizeOne(unsigned int *d_array_final_index, unsigned int *d_array_index, unsigned int *d_array_static_index, unsigned int *d_array_map, unsigned int *d_array_stencil, unsigned int *d_array_first_sort_rank, int size, int originalLength) {

	int threadID = (blockIdx.x * blockDim.x) +  threadIdx.x;

	if(threadID >= size) return;

	d_array_stencil[threadID] = 1;

	if(threadID == 0 && (d_array_map[threadID + 1] == 1)) { 
		d_array_stencil[threadID] = 0; 
	} else if( (threadID == (size-1)) && (d_array_map[threadID] == 1) ) {
		d_array_stencil[threadID] = 0;  
	} else if( (d_array_map[threadID] == 1) && (d_array_map[threadID + 1] == 1)) { 
		d_array_stencil[threadID] = 0; 
	}

	if(d_array_stencil[threadID] == 0) {
		int finalIndex = d_array_index[threadID];
		d_array_final_index[ d_array_static_index[threadID] ] = finalIndex;
		d_array_first_sort_rank[finalIndex] = d_array_static_index[threadID]; 
	}
}

__global__ void createSecondSort(unsigned char *d_original_string_in, unsigned int *d_array_first_sort_rank, unsigned char *d_array_second_sort, unsigned int *d_array_second_sort_rank, unsigned int *d_array_second_sort_index, int secondSortLength) { 

	int threadID = (blockIdx.x * blockDim.x) +  threadIdx.x;

	if(threadID > secondSortLength) return;

	int mult3 = threadID*3;

	d_array_second_sort[threadID] = d_original_string_in[mult3];
	d_array_second_sort_rank[threadID] = d_array_first_sort_rank[mult3+1];
	d_array_second_sort_index[threadID] = mult3;
}

void gpuSetDevice(int devId) { 
	cudaSetDevice(devId);
	return;
}
int gpuBlockSort(UChar *block, UInt32 *order, UInt32 *orderFirstSort, UInt32 *orderSecondSort, UInt32 *orderFirstSortRank, Int32 blockSize, Int32* sortingDepth) { 

	int limit = 64; 
	int length = blockSize;
	int originalLength = blockSize; 


	cudaMalloc((unsigned int **)&global_int_original_string, sizeof(unsigned int)*originalLength);
	cudaMalloc((unsigned int **)&global_first_sort_rank, sizeof(unsigned int)*originalLength);

	unsigned char *d_original_string_in; 
	cudaMalloc((unsigned char **)&d_original_string_in, sizeof(unsigned char)*originalLength);
	cudaMemcpy(d_original_string_in, block, sizeof(unsigned char)*originalLength, cudaMemcpyHostToDevice); 
	
	int numBlocks1 = 1;
	int numThreadsPerBlock1 = originalLength/numBlocks1;

	if(numThreadsPerBlock1 > MAX_THREADS_PER_BLOCK) { 
		numBlocks1 = (int)ceil(originalLength/(float)MAX_THREADS_PER_BLOCK);
		numThreadsPerBlock1 = MAX_THREADS_PER_BLOCK;
	}
	dim3 grid1(numBlocks1, 1, 1);
	dim3 threads1(numThreadsPerBlock1, 1, 1); 


	int firstSortLength =  2*((originalLength-1)/3) + ((originalLength-1)%3);
	int secondSortLength = originalLength - firstSortLength;

	int includeLast = 0;
	if(originalLength % 3 == 1) { 
		includeLast = 1;
		firstSortLength++;
		secondSortLength--;
	}

  	thrust::device_vector<unsigned int> d_stencil(firstSortLength, 0);
	thrust::device_vector<unsigned int> d_index(firstSortLength);
	thrust::device_vector<unsigned int> d_final_index(firstSortLength);

	unsigned int *d_array_index_out = thrust::raw_pointer_cast(&d_index[0]);
	unsigned int *d_array_string_out = thrust::raw_pointer_cast(&d_stencil[0]);

	cudaThreadSynchronize();
	pack4CharsToInt<<<grid1, threads1, 0>>>(d_original_string_in, global_int_original_string, d_array_string_out, d_array_index_out, global_first_sort_rank, originalLength);
	cudaThreadSynchronize();

	if(includeLast == 1) { 
		int lastIndex = originalLength - 1;
		*(d_index.end() - 1) = lastIndex;
		*(d_stencil.end() - 1) =  (((unsigned int)block[lastIndex]) << 24) + 
				  (((unsigned int)block[(lastIndex+1) % length ]) << 16) +  
                                  (((unsigned int)block[(lastIndex+2) % length ]) << 8) + 
				  (((unsigned int)block[(lastIndex+3) % length ]));

	}

	length = firstSortLength;

#ifdef __DEBUG__
	printf("First sorting problem length : %d\n", firstSortLength);
	thrust::device_vector<unsigned int>::iterator d_index_itr = d_index.begin();
	thrust::device_vector<unsigned int>::iterator d_stencil_itr = d_stencil.begin();

	for(d_index_itr = d_index.begin(); d_index_itr != d_index.end(); ++d_index_itr) { 
		std::cout << "( " << *d_index_itr << " , " << *d_stencil_itr << " ) ";
		++d_stencil_itr;
	}
	std::cout << std::endl; 
#endif


	thrust::device_vector<unsigned int> d_string(length); 
	thrust::copy(d_stencil.begin(), d_stencil.begin() + length, d_string.begin());

	thrust::device_vector<unsigned int> d_static_index(length);
	thrust::sequence(d_static_index.begin(), d_static_index.end());

	thrust::device_vector<unsigned int> d_segment(length, 0);
	thrust::device_vector<unsigned int> d_map(length, 0);
        thrust::device_vector<unsigned int> d_integer_arr(length, 0);



	int sequenceCount = 0;

	for(sequenceCount=0; sequenceCount <= limit; sequenceCount+=4) { 
		//Changed to sort by key, this ignores the index sorting

		thrust::sort_by_key(
				thrust::make_zip_iterator( thrust::make_tuple(d_segment.begin(), d_string.begin())),
				thrust::make_zip_iterator( thrust::make_tuple(d_segment.begin() + length, d_string.begin() + length)),
				d_index.begin()
			); 
 
 
		unsigned int *d_array_string = thrust::raw_pointer_cast(&d_string[0]); 
		unsigned int *d_array_index = thrust::raw_pointer_cast(&d_index[0]);
		unsigned int *d_array_segment = thrust::raw_pointer_cast(&d_segment[0]); 
 	  	unsigned int *d_array_static_index = thrust::raw_pointer_cast(&d_static_index[0]); 
          	unsigned int *d_array_map = thrust::raw_pointer_cast(&d_map[0]); 
		unsigned int *d_array_stencil = thrust::raw_pointer_cast(&d_stencil[0]); 
		unsigned int *d_array_final_index = thrust::raw_pointer_cast(&d_final_index[0]);

		int numBlocks = 1;
		int numThreadsPerBlock = length/numBlocks;

		if(numThreadsPerBlock > MAX_THREADS_PER_BLOCK) { 
			numBlocks = (int)ceil(numThreadsPerBlock/(float)MAX_THREADS_PER_BLOCK);
			numThreadsPerBlock = MAX_THREADS_PER_BLOCK;
		}
		dim3 grid(numBlocks, 1, 1);
		dim3 threads(numThreadsPerBlock, 1, 1); 

          	cudaThreadSynchronize();
	
		findSuccessor<<<grid, threads, 0>>>(global_int_original_string, d_array_string, d_array_index, d_array_segment, d_array_stencil, d_array_map, length, originalLength, sequenceCount);
	
	        cudaThreadSynchronize();


	        thrust::copy(d_stencil.begin(), d_stencil.begin() + length, d_string.begin());

	        thrust::inclusive_scan(d_map.begin(),d_map.begin() + length, d_segment.begin());

	        cudaThreadSynchronize();
 
	        eliminateSizeOneKernel1<<<grid, threads, 0>>>( global_int_original_string, d_array_final_index, d_array_index, d_array_static_index, d_array_map, d_array_stencil, global_first_sort_rank, sequenceCount, length, originalLength);
 
                cudaThreadSynchronize();
		
		thrust::exclusive_scan(d_stencil.begin(), d_stencil.begin() + length, d_map.begin());

		thrust::scatter_if(d_segment.begin(), d_segment.begin() + length, d_map.begin(), d_stencil.begin(), d_integer_arr.begin());
		thrust::copy(d_integer_arr.begin(), d_integer_arr.begin() + length, d_segment.begin()); 

		thrust::scatter_if(d_string.begin(), d_string.begin() + length, d_map.begin(), d_stencil.begin(), d_integer_arr.begin()); 
		thrust::copy(d_integer_arr.begin(), d_integer_arr.begin() + length, d_string.begin());


		thrust::scatter_if(d_index.begin(), d_index.begin() + length, d_map.begin(), d_stencil.begin(), d_integer_arr.begin()); 
		thrust::copy(d_integer_arr.begin(), d_integer_arr.begin() + length, d_index.begin()); 

		thrust::scatter_if(d_static_index.begin(), d_static_index.begin() + length, d_map.begin(), d_stencil.begin(), d_integer_arr.begin()); 
		thrust::copy(d_integer_arr.begin(), d_integer_arr.begin() + length, d_static_index.begin()); 

		length = *(d_map.begin() + length - 1) + *(d_stencil.begin() + length - 1); 
	        if(length == 0) {
			 *sortingDepth = sequenceCount;  
             	         break;
	  	} 

	}


	if(length!=0) { 

		int size = length; 
		length = limit*2;
		int offset = limit;  
		for(offset = limit; offset < originalLength; offset+=(length/2)) {  

			thrust::sort(
					thrust::make_zip_iterator( thrust::make_tuple(d_segment.begin(), d_index.begin())),
					thrust::make_zip_iterator( thrust::make_tuple(d_segment.begin() + size, d_index.begin() + size)),
					Bar(offset, length, originalLength)
				    );

			unsigned int *d_array_index = thrust::raw_pointer_cast(&d_index[0]);
			unsigned int *d_array_segment = thrust::raw_pointer_cast(&d_segment[0]); 
			unsigned int *d_array_map = thrust::raw_pointer_cast(&d_map[0]);
			unsigned int *d_array_static_index = thrust::raw_pointer_cast(&d_static_index[0]);  
			unsigned int *d_array_stencil = thrust::raw_pointer_cast(&d_stencil[0]); 
			unsigned int *d_array_final_index = thrust::raw_pointer_cast(&d_final_index[0]);

			int numBlocks = 1;
			int numThreadsPerBlock = size/numBlocks;

			if(numThreadsPerBlock > MAX_THREADS_PER_BLOCK) { 
				numBlocks = (int)ceil(size/(float)MAX_THREADS_PER_BLOCK);
				numThreadsPerBlock = MAX_THREADS_PER_BLOCK;
			}
			dim3 grid(numBlocks, 1, 1);
			dim3 threads(numThreadsPerBlock, 1, 1); 

			cudaThreadSynchronize();

			updateSegments<<<grid, threads, 0>>>(global_int_original_string, d_array_index, d_array_segment, d_array_map, size, offset, length, originalLength);

			cudaThreadSynchronize();

			thrust::inclusive_scan(d_map.begin(), d_map.begin() + size, d_segment.begin());

			cudaThreadSynchronize();

			eliminateSizeOne<<<grid, threads, 0>>>( d_array_final_index, d_array_index, d_array_static_index, d_array_map, d_array_stencil, global_first_sort_rank, size, originalLength);
			cudaThreadSynchronize();

			thrust::exclusive_scan(d_stencil.begin(), d_stencil.begin() + size, d_map.begin());

			thrust::scatter_if(d_segment.begin(), d_segment.begin() + size, d_map.begin(), d_stencil.begin(), d_integer_arr.begin());
			thrust::copy(d_integer_arr.begin(), d_integer_arr.begin() + size, d_segment.begin()); 

			thrust::scatter_if(d_index.begin(), d_index.begin() + size, d_map.begin(), d_stencil.begin(), d_integer_arr.begin()); 
			thrust::copy(d_integer_arr.begin(), d_integer_arr.begin() + size, d_index.begin()); 

			thrust::scatter_if(d_static_index.begin(), d_static_index.begin() + size, d_map.begin(), d_stencil.begin(), d_integer_arr.begin()); 
			thrust::copy(d_integer_arr.begin(), d_integer_arr.begin() + size, d_static_index.begin()); 

			size = *(d_map.begin() + size - 1) + *(d_stencil.begin() + size - 1); 

			if(size == 0) {
				*sortingDepth = offset;
				break;
			}

			length*=2; 
		}

	}

#ifdef __DEBUG__

	printf("First Sort\n");
	thrust::device_vector<unsigned int>::iterator d_final_index_itr = d_final_index.begin();
	for(d_final_index_itr = d_final_index.begin(); d_final_index_itr != d_final_index.end(); ++d_final_index_itr) { 
		std::cout << *d_final_index_itr << " ";
	}
	std::cout << std::endl;
	std::cout << "First Sort Length " << firstSortLength << std::endl;
#endif

	int numBlocks = 1;
	int numThreadsPerBlock = secondSortLength/numBlocks;

	if(numThreadsPerBlock > MAX_THREADS_PER_BLOCK) { 
		numBlocks = (int)ceil(numThreadsPerBlock/(float)MAX_THREADS_PER_BLOCK);
		numThreadsPerBlock = MAX_THREADS_PER_BLOCK;
	}
	dim3 grid(numBlocks, 1, 1);
	dim3 threads(numThreadsPerBlock, 1, 1); 

	thrust::device_vector<unsigned char> d_second_sort(secondSortLength, 0);
	thrust::device_vector<unsigned int> d_second_sort_rank(secondSortLength, 0);
	thrust::device_vector<unsigned int> d_second_sort_index(secondSortLength);

	unsigned char *d_array_second_sort = thrust::raw_pointer_cast(&d_second_sort[0]);
	unsigned int *d_array_second_sort_rank = thrust::raw_pointer_cast(&d_second_sort_rank[0]); 
	unsigned int *d_array_second_sort_index = thrust::raw_pointer_cast(&d_second_sort_index[0]); 

	cudaThreadSynchronize();
	createSecondSort<<<grid, threads, 0>>>(d_original_string_in, global_first_sort_rank, d_array_second_sort, d_array_second_sort_rank, d_array_second_sort_index, secondSortLength);
	cudaThreadSynchronize();

	thrust::sort_by_key(
		thrust::make_zip_iterator( thrust::make_tuple(d_second_sort.begin(), d_second_sort_rank.begin())),
		thrust::make_zip_iterator( thrust::make_tuple(d_second_sort.begin() + secondSortLength, d_second_sort_rank.begin() + secondSortLength)),
		d_second_sort_index.begin()
	);  


#ifdef __DEBUG__
	printf("Second Sort\n");
	thrust::device_vector<unsigned char>::iterator d_second_sort_itr = d_second_sort.begin();
	thrust::device_vector<unsigned int>::iterator d_second_sort_rank_itr = d_second_sort_rank.begin();
	thrust::device_vector<unsigned int>::iterator d_second_sort_index_itr = d_second_sort_index.begin();
	for(d_second_sort_itr = d_second_sort.begin(); d_second_sort_itr != d_second_sort.end(); ++d_second_sort_itr) { 
		std::cout << *d_second_sort_index_itr << " ";
		++d_second_sort_rank_itr;
		++d_second_sort_index_itr;
	}
	std::cout << std::endl;
	std::cout << "Second Sort Length " << secondSortLength << std::endl;
#endif

	thrust::copy(d_final_index.begin(), d_final_index.end(), orderFirstSort);
	thrust::copy(d_second_sort_index.begin(), d_second_sort_index.end(), orderSecondSort);
	cudaMemcpy(orderFirstSortRank, global_first_sort_rank, sizeof(unsigned int)*originalLength, cudaMemcpyDeviceToHost); 
	
	cudaFree(d_original_string_in); 
	cudaFree(global_int_original_string);
	cudaFree(global_first_sort_rank);
	return firstSortLength;
}

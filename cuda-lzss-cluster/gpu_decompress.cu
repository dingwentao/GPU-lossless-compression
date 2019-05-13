/***************************************************************************
 *          Lempel, Ziv, Storer, and Szymanski Encoding and Decoding on CUDA
 *
 *
 ****************************************************************************
 *          CUDA LZSS 
 *   Authors  : Adnan Ozsoy, Martin Swany,Indiana University - Bloomington
 *   Date    : April 11, 2011
 
 ****************************************************************************
 
         Copyright 2011 Adnan Ozsoy, Martin Swany, Indiana University - Bloomington
 
         Licensed under the Apache License, Version 2.0 (the "License");
         you may not use this file except in compliance with the License.
         You may obtain a copy of the License at
 
     http://www.apache.org/licenses/LICENSE-2.0
 
         Unless required by applicable law or agreed to in writing, software
         distributed under the License is distributed on an "AS IS" BASIS,
         WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
         See the License for the specific language governing permissions and
         limitations under the License.
 ****************************************************************************/
 
 /***************************************************************************
 * Code is adopted from below source
 *
 * LZSS: An ANSI C LZss Encoding/Decoding Routine
 * Copyright (C) 2003 by Michael Dipperstein (mdipper@cs.ucsb.edu)
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 *
 ***************************************************************************/



/***************************************************************************
*                             INCLUDED FILES
***************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include "getopt.h"
#include <time.h>
#include "gpu_decompress.h"
#include <pthread.h>
#include <unistd.h>
//#include "cuPrintf.cu"
#include <sys/time.h>


/***************************************************************************
*                             CUDA FILES
***************************************************************************/
#include <assert.h>
#include <cuda.h>
//#include "cuPrintf.cu"
 

/***************************************************************************
*                            GLOBAL VARIABLES
***************************************************************************/
// unsigned char * decompressed_buffer; 

// unsigned char * init_in_d;	
// unsigned char * init_out_d;

/***************************************************************************
*                               PROTOTYPES
***************************************************************************/



void decheckCUDAError(const char *msg)
{
 cudaError_t err = cudaGetLastError();
 if( cudaSuccess != err) 
 {
  fprintf(stderr, "Cuda error: %s: %s.\n", msg, 
 cudaGetErrorString( err) );
  exit(EXIT_FAILURE);
 } 
}


unsigned char * deinitGPUmem(int buf_length)
{
	unsigned char * mem_d;
	
	cudaMalloc((void **) &mem_d, sizeof(char)*buf_length);
	decheckCUDAError("function, initGPUmemIN, mem alloc to gpu");
	
	return mem_d;
}

void dedeleteGPUmem(unsigned char * mem_d)
{
	cudaFree(mem_d);		
}

void deinitGPU()
{
	cudaSetDevice(0);
}


__global__ void DecodeKernel(unsigned char * in_d, unsigned char * out_d, int * error_d, int * sizearr_d, int SIZEBLOCK)
{

	// cyclic buffer sliding window of already read characters //
	unsigned char slidingWindow[WINDOW_SIZE];
	unsigned char uncodedLookahead[MAX_CODED];
	//unsigned char writebuf[8];
	
    int nextChar;                       /* next char in sliding window */
    deencoded_string_t code;              /* offset/length code for string */

    // 8 code flags and encoded strings //
    unsigned char flags, flagsUsed;
    //int nextEncoded;                // index into encodedData //
    //deencoded_string_t matchData;
    int i, c;

     //initialize variables
    flags = 0;
    flagsUsed = 7;
    nextChar = 0;
	
	//long lSize=PCKTSIZE;
	int filepoint=0;
	int wfilepoint=0;

	int bx = blockIdx.x;
	int tx = threadIdx.x; 
	
	int sizeinpckt = 0, startadd = 0;
	startadd = sizearr_d[bx * SIZEBLOCK + tx]; //read the size of the packet
	sizeinpckt = sizearr_d[bx * SIZEBLOCK + tx + 1] -  startadd;
 
	//bigger than a packet hold-compression took more space for that packet
	//REPORT 
	if(sizeinpckt > PCKTSIZE){
		(*error_d)++;
	}

	// ************************************************************************
	//* Fill the sliding window buffer with some known vales.  EncodeLZSS must
	//* use the same values.  If common characters are used, there's an
	//* increased chance of matching to the earlier strings.
	//************************************************************************ /
	for (i = 0; i < WINDOW_SIZE; i++)
	{
		slidingWindow[i] = ' ';
	}

	while (TRUE)
		{
			flags >>= 1;
			flagsUsed++;

			if (flagsUsed == 8)
			{
				// shifted out all the flag bits, read a new flag //
				if (filepoint >= sizeinpckt)
				{
					break;
				}
				c=in_d[startadd + filepoint]; //packet*PCKTSIZE 
				filepoint++;
				flags = c & 0xFF;
				flagsUsed = 0;
			}

			if (flags & 0x01)
			{
				// uncoded character //
				if (filepoint >= sizeinpckt)
				{
					break;
				}
				
				// write out byte and put it in sliding window //
				out_d[bx * SIZEBLOCK * PCKTSIZE + tx*PCKTSIZE+wfilepoint]=in_d[startadd +filepoint];
				wfilepoint++;
				slidingWindow[nextChar] = in_d[startadd +filepoint];
				nextChar = (nextChar + 1) % WINDOW_SIZE;
				filepoint++;
			}
			else 
			{
				// offset and length //
				if (filepoint >= sizeinpckt)
				{
					break;
				}
				code.length=in_d[startadd +filepoint];
				filepoint++;

				if (filepoint >= sizeinpckt)
				{
					break;
				}
				code.offset =in_d[startadd +filepoint];
				filepoint++;
				
				// ****************************************************************
				//* Write out decoded string to file and lookahead.  It would be
				//* nice to write to the sliding window instead of the lookahead,
				////* but we could end up overwriting the matching string with the
				///* new string if abs(offset - next char) < match length.
				//**************************************************************** /
				for (i = 0; i < code.length; i++)
				{
					c = slidingWindow[(code.offset + i) % WINDOW_SIZE];
					out_d[bx * SIZEBLOCK * PCKTSIZE + tx*PCKTSIZE + wfilepoint]=c;
					wfilepoint++;
					uncodedLookahead[i] = c;
				}

				// write out decoded string to sliding window //
				for (i = 0; i < code.length; i++)
				{
					slidingWindow[(nextChar + i) % WINDOW_SIZE] =
						uncodedLookahead[i];
				}

				nextChar = (nextChar + code.length) % WINDOW_SIZE;
			}
		}
		
}
	

int decompression_kernel_wrapper(unsigned char *buffer, int buf_length, int * decomp_length, int compression_type,int wsize, int numthre)
{

	int i,j;
	int numblocks;
	int devID;
	unsigned char * in_d, * out_d;	
	cudaDeviceProp props;
	int numThreads=numthre;

	//get original file size from compressed file
	int origsize=0;
	origsize = buffer[buf_length-6] << 24;
	origsize = origsize  ^ ( buffer[buf_length-5]<< 16);
	origsize = origsize  ^ ( buffer[buf_length-4]<< 8);
	origsize = origsize  ^ ( buffer[buf_length-3]);
	int lSize =  origsize;

	//padsize
	int padsize=0;
	padsize = ( buffer[buf_length-2]<< 8);
	padsize = padsize  ^ ( buffer[buf_length-1]);

	//printf("orig size: %d %d\n",origsize,padsize);
	
	numblocks =lSize / ((numThreads)*PCKTSIZE);
	
    // get number of SMs on this GPU
    cudaGetDevice(&devID);
    cudaGetDeviceProperties(&props, devID);
	
	// put the bottom size headers into an array
	int  * sizearr, * sizearr_d;
	sizearr = (int *)malloc(sizeof(int)*(numThreads*numblocks+1));
	if (sizearr == NULL) {printf ("Memory error"); exit (2);}
	

	sizearr[0]=0;
	int temptot=0;
	int total=0;
	for(i=1, j=0;i<(numThreads*numblocks+1);i++, j=j+2)
	{
		temptot = buffer[buf_length-2*numThreads*numblocks+j-6] << 8;
		temptot = temptot  ^ ( buffer[buf_length-2*numThreads*numblocks+j+1-6]);

		total = total + temptot;
		sizearr[i]=total;	
	}
	

	int * error_c;
	error_c = (int *)malloc(sizeof(int));
	if (error_c==NULL) exit (1);
	
	*error_c = 0;
	
	int * error_d;
	
	//cudaMalloc((void **) &print_d, sizeof(char)*numThreads);
	cudaMalloc((void **) &error_d, sizeof(int));
	
	//cudaMemcpy(print_d, print_cpu, sizeof(char)*numThreads,cudaMemcpyHostToDevice);
	cudaMemcpy(error_d, error_c, sizeof(int),cudaMemcpyHostToDevice);
	decheckCUDAError("mem copy0");

	//Allocate cuda memory
	cudaMalloc((void **) &in_d, sizeof(char)*(buf_length-2*numThreads*numblocks-6));
	cudaMalloc((void **) &out_d, sizeof(char)*lSize);
	cudaMalloc((void **) &sizearr_d, sizeof(int)*(numThreads*numblocks+1));	

	//copy memory to cuda
	cudaMemcpy(in_d, buffer, sizeof(char)*(buf_length-2*numThreads*numblocks-6),cudaMemcpyHostToDevice);
	cudaMemcpy(sizearr_d, sizearr, sizeof(int)*(numThreads*numblocks+1),cudaMemcpyHostToDevice);
	decheckCUDAError("mem copy1");

	//cudaPrintfInit();
		
	//decompression kernel
	DecodeKernel<<< numblocks, numThreads >>>(in_d,out_d,error_d,sizearr_d,numThreads);
	
	
	// Check for any CUDA errors
	decheckCUDAError("kernel invocation");
	//printf(" kernel done\n");

	
	//copy memory back
	cudaMemcpy(buffer/*decompressed_buffer*/, out_d, sizeof(char)*(lSize-padsize), cudaMemcpyDeviceToHost);
	//cudaMemcpy(print_cpu, print_d, sizeof(char)*numThreads, cudaMemcpyDeviceToHost);
	decheckCUDAError("mem copy2");

	cudaMemcpy(error_c, error_d, sizeof(int),cudaMemcpyDeviceToHost);

	
	decheckCUDAError("mem copy3");
	
	if(*error_c !=0)
		printf("Compression took more space for some packets !!! code: %d \n", *error_c);

	* decomp_length = lSize-padsize;
		
	//free cuda memory
	cudaFree(in_d);
	//cudaFree(print_d);
	cudaFree(error_d);
	cudaFree(out_d);
	cudaFree(sizearr_d);
	//free(print_cpu);
	free(error_c);
	
	return 1;
}





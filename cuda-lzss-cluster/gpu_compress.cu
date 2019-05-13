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
#include "gpu_compress.h"
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
//unsigned char * decompressed_buffer; 

//unsigned char * init_in_d;	
//unsigned char * init_out_d;

texture<unsigned char, 1, cudaReadModeElementType> in_d_tex;

cudaStream_t * streams;
int instreams = 16;
int nstreams = 4*instreams;


/***************************************************************************
*                               PROTOTYPES
***************************************************************************/


/****************************************************************************
*   Function   : FindMatch
*   Description: This function will search through the slidingWindow
*                dictionary for the longest sequence matching the MAX_CODED
*                long string stored in uncodedLookahed.
*   Parameters : windowHead - head of sliding window
*                uncodedHead - head of uncoded lookahead buffer
*   Effects    : NONE
*   Returned   : The sliding window index where the match starts and the
*                length of the match.  If there is no match a length of
*                zero will be returned.
****************************************************************************/
__device__ encoded_string_t FindMatch(int windowHead, int uncodedHead, unsigned char* slidingWindow, unsigned char* uncodedLookahead, \
		int tx, int bx, int wfilepoint, int lastcheck, int loadcounter)
{
    encoded_string_t matchData;
    int i, j;
	int maxcheck;
	int matchingState=0;
	int loop=0;
	
	matchData.length = 1; // make it 1 in the 0 case, it will be returned as 1, 0 gives problems
	matchData.offset = 1; // make it 1 in the 0 case, it will be returned as 1, 0 gives problems
    i = windowHead ;  // start at the beginning of the sliding window //
    j = 0; //counter for matchings

	
	//if(lastcheck) 
		maxcheck = MAX_CODED - tx*lastcheck;
	//else
	//	maxcheck = MAX_CODED;
	
	int tempi=0;
	while (loop<WINDOW_SIZE)
	{
		if (slidingWindow[i] == uncodedLookahead[(uncodedHead+j)% (WINDOW_SIZE+MAX_CODED)])
		{
			j++;
			matchingState=1;		
		}
		else
		{
			if(matchingState && j > matchData.length)
			{
				matchData.length = j;
				tempi=i-j;
				if(tempi<0)
					tempi+=WINDOW_SIZE+MAX_CODED;
				matchData.offset = tempi;
			}
			
			j=0;
			matchingState=0;		
		}
	
		i = (i + 1) % (WINDOW_SIZE+MAX_CODED);
		loop++;	
		if (loop >= maxcheck-1)
		{
			/// we wrapped around ///
			loop = WINDOW_SIZE; //break;
		}
	}
	
	if(j > matchData.length && matchingState )
	{
		matchData.length = j;
		tempi=i-j;
		if(tempi<0)
			tempi+=WINDOW_SIZE+MAX_CODED;

		matchData.offset = tempi;
	}
		
	
    return matchData;
}

void checkCUDAError(const char *msg)
{
 cudaError_t err = cudaGetLastError();
 if( cudaSuccess != err) 
 {
  fprintf(stderr, "Cuda error: %s: %s.\n", msg, 
 cudaGetErrorString( err) );
  exit(EXIT_FAILURE);
 } 
}


__global__ void EncodeKernel(unsigned char * in_d, unsigned char * out_d, int SIZEBLOCK)
{


	/* cyclic buffer sliding window of already read characters */
	__shared__ unsigned char slidingWindow[WINDOW_SIZE+(MAX_CODED)];
	__shared__ unsigned char uncodedLookahead[MAX_CODED*2];
	__shared__ unsigned char encodedData[MAX_CODED*2];
    encoded_string_t matchData;

	int windowHead, uncodedHead;    // head of sliding window and lookahead //
	int filepoint;			//file index pointer for reading
	int wfilepoint;			//file index pointer for writing
	int lastcheck;			//flag for last run of the packet
	int loadcounter=0;
	
	int bx = blockIdx.x;
	int tx = threadIdx.x; 
	
	
   //***********************************************************************
   // * Fill the sliding window buffer with some known values.  DecodeLZSS must
   // * use the same values.  If common characters are used, there's an
   // * increased chance of matching to the earlier strings.
   // *********************************************************************** //

	slidingWindow[tx] = ' ';
	windowHead = tx;
	uncodedHead = tx;	
	filepoint=0;
	wfilepoint=0;
	lastcheck=0;
	
	__syncthreads();

	
	//***********************************************************************
	//* Copy MAX_CODED bytes from the input file into the uncoded lookahead
	//* buffer.
	//*********************************************************************** //
  
	//uncodedLookahead[tx] = tex1Dfetch(in_d_tex, bx * PCKTSIZE + tx); //in_d[bx * PCKTSIZE + tx];
	uncodedLookahead[tx] = in_d[bx * PCKTSIZE + tx];
	filepoint+=MAX_CODED;
	
	slidingWindow[ (windowHead + WINDOW_SIZE ) % (WINDOW_SIZE + MAX_CODED) ] = uncodedLookahead[uncodedHead];
	//tex1Dfetch(in_d_tex, bx * PCKTSIZE + tx);//uncodedLookahead[uncodedHead];
	
	__syncthreads(); 
	
	//uncodedLookahead[MAX_CODED+tx] = tex1Dfetch(in_d_tex, bx * PCKTSIZE + filepoint + tx); //in_d[bx * PCKTSIZE + filepoint + tx];
	uncodedLookahead[MAX_CODED+tx] = in_d[bx * PCKTSIZE + filepoint + tx];
	filepoint+=MAX_CODED;
	

	
	__syncthreads();
	
    loadcounter++;
	// Look for matching string in sliding window //	
	matchData = FindMatch(windowHead, uncodedHead,slidingWindow,uncodedLookahead,  tx, bx, 0, 0,loadcounter);
	__syncthreads();  
	
	// now encoded the rest of the file until an EOF is read //
	while ((filepoint) <= PCKTSIZE && !lastcheck)
	{		
	
		
		
		if (matchData.length >= MAX_CODED)
        	{
            		// garbage beyond last data happened to extend match length //
            		matchData.length = MAX_CODED-1;
      		}

		if (matchData.length <= MAX_UNCODED)
		{
			// not long enough match.  write uncoded byte //
			matchData.length = 1;   // set to 1 for 1 byte uncoded //
			encodedData[tx*2] = 1;
			encodedData[tx*2 + 1] = uncodedLookahead[uncodedHead];
		}
		else if(matchData.length > MAX_UNCODED)
		{	
			// match length > MAX_UNCODED.  Encode as offset and length. //
			encodedData[tx*2] = (unsigned char)matchData.length;
			encodedData[tx*2+1] = (unsigned char)matchData.offset;			
		}

			
		//write out the encoded data into output
		out_d[bx * PCKTSIZE*2 + wfilepoint + tx*2] = encodedData[tx*2];
		out_d[bx * PCKTSIZE*2 + wfilepoint + tx*2 + 1] = encodedData[tx*2+1];
		
		//update written pointer and heads
		wfilepoint = wfilepoint + MAX_CODED*2;
		
		windowHead = (windowHead + MAX_CODED) % (WINDOW_SIZE+MAX_CODED);
		uncodedHead = (uncodedHead + MAX_CODED) % (MAX_CODED*2);
		
		__syncthreads(); 	

				
		//if(lastcheck==1)
		//{
		//	break;			
		//}	
		
		//if(!lastcheck)
		{
			if(filepoint<PCKTSIZE){
				//uncodedLookahead[(uncodedHead+ MAX_CODED)% (MAX_CODED*2)] = tex1Dfetch(in_d_tex, bx * PCKTSIZE + filepoint + tx);
				uncodedLookahead[(uncodedHead+ MAX_CODED)% (MAX_CODED*2)] = in_d[bx * PCKTSIZE + filepoint + tx];
				filepoint+=MAX_CODED;
				
				//find the location for the thread specific view of window
				slidingWindow[ (windowHead + WINDOW_SIZE ) % (WINDOW_SIZE + MAX_CODED) ] = uncodedLookahead[uncodedHead];
				//__syncthreads(); 	
			}
			else{
				lastcheck++;				
				slidingWindow[(windowHead + MAX_CODED ) % (WINDOW_SIZE+MAX_CODED)] = '^';		
			}
			__syncthreads(); 	
			
			loadcounter++;
			matchData = FindMatch(windowHead, uncodedHead,slidingWindow,uncodedLookahead,tx,bx, wfilepoint, lastcheck,loadcounter);
		}
		
	} //while
	
		if(lastcheck==1)
		{
			if(matchData.length > (MAX_CODED - tx))
				matchData.length = MAX_CODED - tx;
		}
		
		if (matchData.length >= MAX_CODED)
        	{
            	// garbage beyond last data happened to extend match length //
            	matchData.length = MAX_CODED-1;
      		}

		if (matchData.length <= MAX_UNCODED)
		{
			// not long enough match.  write uncoded byte //
			matchData.length = 1;   // set to 1 for 1 byte uncoded //
			encodedData[tx*2] = 1;
			encodedData[tx*2 + 1] = uncodedLookahead[uncodedHead];
		}
		else if(matchData.length > MAX_UNCODED)
		{	
			// match length > MAX_UNCODED.  Encode as offset and length. //
			encodedData[tx*2] = (unsigned char)matchData.length;
			encodedData[tx*2+1] = (unsigned char)matchData.offset;			
		}

			
		//write out the encoded data into output
		out_d[bx * PCKTSIZE*2 + wfilepoint + tx*2] = encodedData[tx*2];
		out_d[bx * PCKTSIZE*2 + wfilepoint + tx*2 + 1] = encodedData[tx*2+1];
		
		//update written pointer and heads
		wfilepoint = wfilepoint + MAX_CODED*2;
		
		windowHead = (windowHead + MAX_CODED) % (WINDOW_SIZE+MAX_CODED);
		uncodedHead = (uncodedHead + MAX_CODED) % (MAX_CODED*2);
		
}

unsigned char * initGPUmem(int buf_length)
{
	unsigned char * mem_d;
	
	cudaMalloc((void **) &mem_d, sizeof(char)*buf_length);
	checkCUDAError("function, initGPUmemIN, mem alloc to gpu");
	
	return mem_d;
}

unsigned char * initCPUmem(int buf_length)
{
	unsigned char * mem_d;
	
	cudaMallocHost((void **) &mem_d, sizeof(char)*buf_length);
	checkCUDAError("function, initCPUmemIN, mem alloc to cpu");
	
	return mem_d;
}

void deleteGPUmem(unsigned char * mem_d)
{
	cudaFree(mem_d);		
}
void deleteCPUmem(unsigned char * mem_d)
{
	cudaFreeHost(mem_d);		
	checkCUDAError("deleteCPUmem func,cudaFreeHost");

}

void deleteGPUStreams()
{
	for (int i = 0; i < nstreams; ++i) 
	{
		cudaStreamDestroy(streams[i]);	
		checkCUDAError("deleteCPUmem func, cudaStreamDestroy" + i);
	}
}

void initGPU()
{
	//cudaDeviceReset();
	cudaSetDevice(0);
	//cudaSetDeviceFlags(cudaDeviceScheduleAuto);
	checkCUDAError("initialize GPU");
	streams = (cudaStream_t*) malloc(nstreams * sizeof(cudaStream_t));
    for(int i = 0; i < nstreams; i++) {
        cudaStreamCreate(&(streams[i]));
		checkCUDAError("streams created");
    }	
}

void resetGPU()
{
	cudaDeviceReset();
}

int streams_in_GPU(){

	return true;
}

int onestream_finish_GPU(int index)
{
	//cudaStreamSynchronize(streams[(index+1)*instreams -1]);
	int check = (index+1)*instreams-1;
	if (check == instreams * nstreams)
		check = check -1;
	while(cudaStreamQuery(streams[check])!=cudaSuccess);
	checkCUDAError("cuda stream sync");
	return true;
}

int compression_kernel_wrapper(unsigned char *buffer, int buf_length, unsigned char * bufferout, int compression_type,int wsize,\
								int numthre, int noop,int index,unsigned char * in_d,unsigned char * out_d)
{

	int numThreads = numthre;
	int numblocks = (buf_length / (PCKTSIZE*instreams)) + (((buf_length % (PCKTSIZE*instreams))>0)?1:0);
	int i=0;

	cudaFuncSetCacheConfig(EncodeKernel, cudaFuncCachePreferL1);//cudaFuncCachePreferShared);

	for(i = 0; i < instreams; i++)
	{
		//copy memory to cuda device
		cudaMemcpyAsync(in_d+ i * (buf_length / instreams), buffer+ i * (buf_length / instreams), \
						sizeof(char)*(buf_length / instreams),cudaMemcpyHostToDevice, streams[index*instreams + i]);
		checkCUDAError("mem copy to gpu");
	}
	
    for(i = 0; i < instreams; i++)
	{
		EncodeKernel<<< numblocks, numThreads, 0, streams[index*instreams + i]>>>(in_d + i * (buf_length / instreams),\
						out_d + 2 * i * (buf_length / instreams),numThreads);
		checkCUDAError("kernel invocation");   // Check for any CUDA errors
	}
	//copy memory back
	for(i = 0; i < instreams; i++)
	{	
		cudaMemcpyAsync(bufferout + 2 * i * (buf_length / instreams), out_d + 2 * i * (buf_length / instreams),\
						sizeof(char)*(buf_length / instreams)*2, cudaMemcpyDeviceToHost, streams[index*instreams + i]);
		checkCUDAError("mem copy back");
	}
	
		
	return 1;
}

void *aftercomp (void *q)
{
	aftercompdata_t * data=(aftercompdata_t *)q;
	
	int i=0, j=0, k=0, m=0, temptot=0, tempj=0;
	int finish=0;
	unsigned char flags =0;
	unsigned char flagPos = 0x01;

	unsigned char holdbuf[16];
	int holdbufcount=0;

	int morecounter=0;	
	
	//reset the flags again
	flagPos = 0x01;
	flags =0;
	temptot=0;		
	holdbufcount=0;

	unsigned char * bufferout = data->bufferout;
	unsigned char * buffer = data->buffer;
	int * header = data->header;
	int buf_length = data->buf_length;
	
	i = (data->tid)*((buf_length*2)/(data->numts));
	j = (data->tid)*((buf_length)/(data->numts));
	k = (data->tid)*(buf_length/(PCKTSIZE*data->numts));
	finish = (data->tid + 1)*((buf_length)/(data->numts));

	while(i<(finish*2))
	{
		if (j>finish) { 
			printf("compression took more, size is %d!!! \n",j); 
			data->comptookmore = 1;
			break;
		}
		
		temptot = bufferout[i];
		if(temptot == 1) //if no matching
		{
			flags |= flagPos;       // mark with uncoded byte flag //
			holdbuf[holdbufcount]=bufferout[i+1];
			holdbufcount++;
			i=i+2;
		}		
		else //if there is mathcing
		{
			holdbuf[holdbufcount]=temptot;
			holdbufcount++;
			holdbuf[holdbufcount]=bufferout[i+1];
			holdbufcount++;
			i=i+(temptot*2);
		}
					
		if (flagPos == 0x80) //if we have looked at 8 characters that fills the flag holder
		{
			buffer[j] = flags;
			j++;
			
			for(m=0;m<holdbufcount;m++){
				buffer[j] = holdbuf[m];
				j++;
			}
			
			// reset encoded data buffer //
			flags = 0;
			flagPos = 0x01;
			holdbufcount=0;
		}
		else
		{
			// we don't have 8 code flags yet, use next bit for next flag //
			flagPos <<= 1;
		}

		// for each packet with the size of 4096 bytes
		if(i%8192 == 0 && i>0){ //PCKTSIZE*2
			if(holdbufcount>0){
				buffer[j] = flags;
				j++;
				
				for(m=0;m<holdbufcount;m++){
				buffer[j] = holdbuf[m];
				j++;
				}
				holdbufcount=0;
			}
			
			flags = 0;
			flagPos = 0x01;			
			if((j-tempj) >= PCKTSIZE){
					morecounter++;
					//compression took more, so just write the file without compression info
				}					

			header[k]=j-tempj;
			tempj=j;
			k++;
		}
	}	
	data->newlen = j - (data->tid)*((buf_length)/(data->numts)) ;
	
	return 0;
}


int aftercompression_wrapper(unsigned char * buffer, int buf_length, unsigned char * bufferout, int * comp_length)
{
		
	int comptookmore = 0;

	//struct timeval t1_start,t1_end;
	//double alltime;

	//gettimeofday(&t1_start,0);
	// allocate memory to contain the header of the file:
	int * header;
	header = (int *)malloc (sizeof(int)*(buf_length/PCKTSIZE));
	if (header == NULL) {printf ("Memory error, header"); exit (2);}	

	pthread_t afcomp[NWORKERS];
	aftercompdata_t data[NWORKERS];

	int l=0;
	
	for(l=0;l<NWORKERS;l++)
	{
		data[l].tid=l;
		data[l].header=header;     /* offset to start of longest match */
		data[l].buffer=buffer;
		data[l].buf_length=buf_length;
		data[l].bufferout=bufferout;
		data[l].numts = NWORKERS;
		data[l].comptookmore=0;
		data[l].newlen=0;

		pthread_create (&afcomp[l], NULL, &aftercomp, &data[l]);
		
	}


	int i=0, j=0, k=0;//, m=0, temptot=0, tempj=0;
	void *status;

	for(l=0;l<NWORKERS;l++){
		pthread_join( afcomp[l], &status);
		comptookmore += data[l].comptookmore;
		if(l!=0)
		{
			for(i=0;i<data[l].newlen;i++)
			{
				buffer[j+i]=buffer[(l*(buf_length/NWORKERS))+i];
			}
		}	
		j+=data[l].newlen;
	}

	k=(buf_length/PCKTSIZE);
			
	if(!comptookmore){
		
		//Add header to buffer
		unsigned char cc;
		for(i=0;i<k;i++)
		{
			cc = (unsigned char)(header[i]>>8);
			buffer[j]=cc;
			j++;
			cc=(unsigned char)header[i];
			buffer[j]=cc;
			j++;
		}
		
		//Add total size
		cc = (unsigned char)(buf_length>>24);
		buffer[j]=cc;
		j++;
		cc = (unsigned char)(buf_length>>16);
		buffer[j]=cc;
		j++;
		cc = (unsigned char)(buf_length>>8);
		buffer[j]=cc;
		j++;
		cc=(unsigned char)buf_length;
		buffer[j]=cc;
		j++;
		
		//Add pad size
		int paddingsize = 0;
		cc = (unsigned char)(paddingsize>>8);
		buffer[j]=cc;
		j++;
		cc=(unsigned char)paddingsize;
		buffer[j]=cc;
		j++;
		

	}
	if(comptookmore!=0)
		return 0;

	if(j>buf_length)
		printf("compression TOOK more!!! %d\n",j);
	
	*comp_length = j;
	free(header);
	
	return 1;


}
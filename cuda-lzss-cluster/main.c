/***************************************************************************
 *  Lempel, Ziv, Storer, and Szymanski Encoding and Decoding on CUDA
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
 
 
#include <pthread.h>
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include "culzss.h"
#include <sys/time.h>
#include <string.h>
#include <signal.h>
#include "getopt.h"
#include "decompression.h"

#define MINSIZE 65536 //335544321 // //1MB size for minimum process
#define BUFSIZE 1048576//65536 //1MB size for buffers

struct timeval tall_start,tall_end;
double alltime_signal;
int loopcount=0;
queue *fifo;
char * inputfilename;
char * outputfilename;
int maxiters=0;
int padding=0;
int numbls=0;
int blsize=0;
int totalsize=0;
unsigned int * bookkeeping;
int decomp=0;
int buffersize =0;

// Define the function to be called when ctrl-c (SIGINT) signal is sent to process
void signal_callback_handler(int signum)
{
	gettimeofday(&tall_end,0);
	alltime_signal = (tall_end.tv_sec-tall_start.tv_sec) + (tall_end.tv_usec - tall_start.tv_usec)/1000000.0;
	printf("\tAll the time took:\t%f \n", alltime_signal);
	int sizeinmb = blsize / (1024*1024);
	printf("\tThroughput for %d runs(proc %d)  of %dMB is :\t%lfMbps \n", getloopcount(),loopcount, sizeinmb, (getloopcount()*blsize*8)/alltime_signal);
	
	printf("Caught signal %d\n",signum);
   // Cleanup and close up stuff here
	queueDelete (fifo);
   // Terminate program
   exit(signum);
}


void *producer (void *q)
{
	struct timeval t1_start,t1_end;
	double alltime;
	
    FILE *inFile;//, *outFile, *decFile;  /* input & output files */
	inFile = NULL;
	queue *fifo;
	int i;	

	//read file into memory
	fifo = (queue *)q;
	if ((inFile = fopen(inputfilename, "rb")) == NULL){
		printf ("Memory error, temp"); exit (2);
	}		
	
	for (i = 0; i < maxiters; i++) {
		
		gettimeofday(&t1_start,0);	
	
		pthread_mutex_lock (fifo->mut);
		while (fifo->ledger[fifo->headPG]!=0) {
			//printf ("producer: queue FULL.\n");
			pthread_cond_wait (fifo->sent, fifo->mut);
		}
		
		int result = fread (fifo->buf[fifo->headPG],1,blsize,inFile);
		if (result != blsize ) 
		{
			if(i!=maxiters-1)
			{	
				printf ("Reading error1, expected size %d, read size %d ", blsize,result); 
				exit (3);
			}	
		}
		
		fifo->ledger[fifo->headPG]++;
		fifo->headPG++;
		if (fifo->headPG == numbls)
			fifo->headPG = 0;

		pthread_mutex_unlock (fifo->mut);
		pthread_cond_signal (fifo->produced);
		
		
		gettimeofday(&t1_end,0);
		alltime = (t1_end.tv_sec-t1_start.tv_sec) + (t1_end.tv_usec - t1_start.tv_usec)/1000000.0;
	}
	fclose(inFile);
	
	return (NULL);
}

int main (int argc, char* argv[]) 
{
	// Register signal and signal handler
	signal(SIGINT, signal_callback_handler);
	double alltime;
	
	int opt;
	numbls = 4;
	maxiters = 0;
	
	/* parse command line */
	while ((opt = getopt(argc, argv, "i:o:d:h:")) != -1)
    {
      switch(opt)
        {
	
		case 'i':       /* input file name */
			inputfilename = optarg;
			break;

		case 'o':       /* output file name */
			outputfilename = optarg;
			break;

		case 'd':       /* decompression */
                decomp = atoi(optarg);
                break;

		case 'h':       /* help */
                printf(" Usage for compression: ./main -i {inputfile} -o {outputfile}\n");
                printf(" Usage for decompression: ./main -d 1 -i {inputfile} -o {outputfile}\n");
                return;
				
		// case 'b':       /* buf size */
                // buffersize = atoi(optarg);
                // break;				
        }
    }
    FILE *filein;//, *outFile, *decFile;  /* input & output files */
	filein = NULL;

	if(inputfilename==NULL)
    {
		printf(" Usage for compression: ./main -i {inputfile} -o {outputfile}\n");
        printf(" Usage for decompression: ./main -d 1 -i {inputfile} -o {outputfile}\n");
        return;		
	}
	
	if ((filein = fopen(inputfilename, "rb")) == NULL){
		printf ("File reading error"); exit (2);
	}	
	fseek(filein , 0 , SEEK_END);
	totalsize = ftell (filein);
	fclose(filein);	
	
	//if (buffersize==0)
	buffersize = BUFSIZE;
	
	if(decomp)
	{
		printf("Doing decompression\n");
		gettimeofday(&tall_start,0);	

		decompression(inputfilename,buffersize,outputfilename);
		
		gettimeofday(&tall_end,0);
		alltime = (tall_end.tv_sec-tall_start.tv_sec) + (tall_end.tv_usec - tall_start.tv_usec)/1000000.0;
		printf("\tAll the time took:\t%f \n", alltime);
		
		int sizeinmb= totalsize / (1024*1024);
		
		printf("\tThroughput for  %dMB is :\t%lfMbps \n", sizeinmb, (sizeinmb*8)/alltime);
		return;
	}
	

	printf("file size:%d",totalsize);
	//decide buf sizes
	
	if(totalsize < BUFSIZE)
	{
		printf(", too small to benefit from GPU\n");
		return;
	}
	else
	
	{
		maxiters = totalsize / buffersize + (((totalsize % buffersize) > 0)?1:0);
		padding = totalsize % buffersize;
		padding = (padding)?(buffersize-padding):0;
		blsize = buffersize;
	}
	printf(" eachblock_size:%d num_of_blocks:%d padding:%d \n",blsize,maxiters, padding);
	
	bookkeeping = (unsigned int * )malloc(sizeof(int)*(maxiters+2)); //# of blocks, each block size, padding size 
	bookkeeping[0] = maxiters;
	bookkeeping[1] = padding;
	
	gettimeofday(&tall_start,0);	

	
	pthread_t pro;
	
	fifo = queueInit (maxiters,numbls,blsize);
	if (fifo ==  NULL) {
		fprintf (stderr, "main: Queue Init failed.\n");
		exit (1);
	}	
	
	//init compression threads
	init_compression(fifo,maxiters,numbls,blsize,outputfilename,bookkeeping);

	//create producer
	pthread_create (&pro, NULL, producer, fifo);

	//join all 
	join_comp_threads();
	//join producer
	pthread_join (pro, NULL);
	queueDelete (fifo);

	gettimeofday(&tall_end,0);
	alltime = (tall_end.tv_sec-tall_start.tv_sec) + (tall_end.tv_usec - tall_start.tv_usec)/1000000.0;
	printf("\tAll the time took:\t%f \n", alltime);
	int sizeinmb= totalsize / (1024*1024);
	printf("\tThroughput for %d runs of %dMB is :\t%lfMbps \n", maxiters,sizeinmb, (sizeinmb*8)/alltime);

	free(bookkeeping);
	//exit
	return 0;

} 




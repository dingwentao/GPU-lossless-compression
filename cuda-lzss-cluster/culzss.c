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

 #include <pthread.h>
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include "culzss.h"
#include <sys/time.h>
#include <string.h>

pthread_t congpu, constr, concpu, consend;

int loopnum=0;
int maxiterations=0;
int numblocks=0;
int blocksize=0;
char * outputfilename;
unsigned int * bookkeeping;

int exit_signal = 0;


int getloopcount(){
	return loopnum;
}

void *gpu_consumer (void *q)
{

	struct timeval t1_start,t1_end,t2_start,t2_end;
	double time_d, alltime;

	queue *fifo;
	int i, d;
	int success=0;
	fifo = (queue *)q;
	int comp_length=0;
	
	fifo->in_d = initGPUmem((int)blocksize);
	fifo->out_d = initGPUmem((int)blocksize*2);
	
	
	for (i = 0; i < maxiterations; i++) {
		
		if(exit_signal){
			exit_signal++;
			break;
		}
		
		gettimeofday(&t1_start,0);	
	
		pthread_mutex_lock (fifo->mut);
		while (fifo->ledger[fifo->headGC]!=1) 
		{
			//printf ("gpu consumer: queue EMPTY.\n");
			pthread_cond_wait (fifo->produced, fifo->mut);
		}
		pthread_mutex_unlock (fifo->mut);
		
		gettimeofday(&t2_start,0);	
		
		success=compression_kernel_wrapper(fifo->buf[fifo->headGC], blocksize, fifo->bufout[fifo->headGC], 
										0, 0, 128, 0,fifo->headGC, fifo->in_d, fifo->out_d);
		if(!success){
			printf("Compression failed. Success %d\n",success);
		}	
		
		gettimeofday(&t2_end,0);
		time_d = (t2_end.tv_sec-t2_start.tv_sec) + (t2_end.tv_usec - t2_start.tv_usec)/1000000.0;
		//printf("GPU kernel took:\t%f \t", time_d);
				
		pthread_mutex_lock (fifo->mut);
		fifo->ledger[fifo->headGC]++;
		fifo->headGC++;
		if (fifo->headGC == numblocks)
			fifo->headGC = 0;

		pthread_mutex_unlock (fifo->mut);
		
		pthread_cond_signal (fifo->compressed);
		
		gettimeofday(&t1_end,0);
		alltime = (t1_end.tv_sec-t1_start.tv_sec) + (t1_end.tv_usec - t1_start.tv_usec)/1000000.0;
		//printf("GPU whole took:\t%f \n", alltime);
	}
	
	deleteGPUmem(fifo->in_d);
	deleteGPUmem(fifo->out_d);
	
	return (NULL);
}


void *cpu_consumer (void *q)
{

	struct timeval t1_start,t1_end,t2_start,t2_end;
	double time_d, alltime;

	int i;
	int success=0;
	queue *fifo;	
	fifo = (queue *)q;
	int comp_length=0;
	unsigned char * bckpbuf;
	bckpbuf = (unsigned char *)malloc(sizeof(unsigned char)*blocksize);
	
	for (i = 0; i < maxiterations; i++) {

		if(exit_signal){
			exit_signal++;
			break;
		}
		gettimeofday(&t1_start,0);	

		pthread_mutex_lock (fifo->mut);
		while (fifo->ledger[fifo->headCS]!=2) {
			//printf ("cpu consumer: queue EMPTY.\n");
			pthread_mutex_unlock (fifo->mut);
			pthread_mutex_lock (fifo->mut);
		}
		pthread_mutex_unlock (fifo->mut);
	
		onestream_finish_GPU(fifo->headCS);

		gettimeofday(&t2_start,0);	


		memcpy (bckpbuf, fifo->buf[fifo->headCS], blocksize);
		success=aftercompression_wrapper(fifo->buf[fifo->headCS], blocksize, fifo->bufout[fifo->headCS], &comp_length);
		if(!success){
			printf("After Compression failed. Success %d return size %d\n",success,comp_length);
			fifo->outsize[fifo->headCS] = 0;
			memcpy (fifo->buf[fifo->headCS],bckpbuf,  blocksize);
		}	
		else
			fifo->outsize[fifo->headCS] = comp_length;
		

		gettimeofday(&t2_end,0);
		time_d = (t2_end.tv_sec-t2_start.tv_sec) + (t2_end.tv_usec - t2_start.tv_usec)/1000000.0;
		
		pthread_mutex_lock (fifo->mut);
		fifo->ledger[fifo->headCS]++;//=0;
		fifo->headCS++;
		if (fifo->headCS == numblocks)
			fifo->headCS = 0;

		pthread_mutex_unlock (fifo->mut);
		
		pthread_cond_signal (fifo->sendready);
		gettimeofday(&t1_end,0);
		alltime = (t1_end.tv_sec-t1_start.tv_sec) + (t1_end.tv_usec - t1_start.tv_usec)/1000000.0;
	}
	return (NULL);
}

void *cpu_sender (void *q)
{
	struct timeval t1_start,t1_end;//,t2_start,t2_end;
	double  alltime;//time_d,
	FILE *outFile;

	int i;
	int success=0;
	queue *fifo;	
	fifo = (queue *)q;
	if(outputfilename!=NULL)
		outFile = fopen(outputfilename, "wb");
	else
		outFile = fopen("compressed.dat", "wb");
	int size=0;

	fwrite(bookkeeping, sizeof(unsigned int), maxiterations+2, outFile);
		
	for (i = 0; i < maxiterations; i++) 
	{
		if(exit_signal){
			exit_signal++;
			break;
		}
		gettimeofday(&t1_start,0);	

		pthread_mutex_lock (fifo->mut);
		while (fifo->ledger[fifo->headSP]!=3) {
			//printf ("cpu_sender: queue EMPTY.\n");
			pthread_cond_wait (fifo->sendready, fifo->mut);
		}
		pthread_mutex_unlock (fifo->mut);
		
			
		pthread_mutex_lock (fifo->mut);

		size = fifo->outsize[fifo->headSP];
		if(size == 0)
			size = blocksize;
		bookkeeping[i + 2] = size + ((i==0)?0:bookkeeping[i+1]);

		fwrite(fifo->buf[fifo->headSP], size, 1, outFile);
		
		//gettimeofday(&t2_end,0);
		//time_d = (t2_end.tv_sec-t2_start.tv_sec) + (t2_end.tv_usec - t2_start.tv_usec)/1000000.0;
		
		fifo->ledger[fifo->headSP]=0;
		fifo->headSP++;
		if (fifo->headSP == numblocks)
			fifo->headSP = 0;

		pthread_mutex_unlock (fifo->mut);
		
		pthread_cond_signal (fifo->sent);

		gettimeofday(&t1_end,0);
		alltime = (t1_end.tv_sec-t1_start.tv_sec) + (t1_end.tv_usec - t1_start.tv_usec)/1000000.0;
		loopnum++;
	}
	fseek ( outFile , 0 , SEEK_SET );
	fwrite(bookkeeping, sizeof(unsigned int), maxiterations+2, outFile);

		
	fclose(outFile);
	return (NULL);
}



queue *queueInit (int maxit,int numb,int bsize)
{
	queue *q;
	maxiterations=maxit;
	numblocks=numb;
	blocksize=bsize;
	
	q = (queue *)malloc (sizeof (queue));
	if (q == NULL) return (NULL);

	int i;
	//alloc bufs
	unsigned char ** buffer;
	unsigned char ** bufferout;
	
	
	/*  allocate storage for an array of pointers */
	buffer = (unsigned char **)malloc((numblocks) * sizeof(unsigned char *));
	if (buffer == NULL) {
		printf("Error: malloc could not allocate buffer\n");
		return;
	}
	bufferout = (unsigned char **)malloc((numblocks) * sizeof(unsigned char *));
	if (bufferout == NULL) {
		printf("Error: malloc could not allocate bufferout\n");
		return;
	}
	  
	/* for each pointer, allocate storage for an array of chars */
	for (i = 0; i < (numblocks); i++) {
		//buffer[i] = (unsigned char *)malloc(blocksize * sizeof(unsigned char));
		buffer[i] = (unsigned char *)initCPUmem(blocksize * sizeof(unsigned char));
		if (buffer[i] == NULL) {printf ("Memory error, buffer"); exit (2);}
	}
	for (i = 0; i < (numblocks); i++) {
		//bufferout[i] = (unsigned char *)malloc(blocksize * 2 * sizeof(unsigned char));
		bufferout[i] = (unsigned char *)initCPUmem(blocksize * 2 * sizeof(unsigned char));
		if (bufferout[i] == NULL) {printf ("Memory error, bufferout"); exit (2);}
	}
	
	q->buf = buffer;
	q->bufout = bufferout;
	
	q->headPG = 0;
	q->headGC = 0;
	q->headCS = 0;
	q->headSP = 0;
	
	q->outsize = (int *)malloc(sizeof(int)*numblocks);
	
	q->ledger = (int *)malloc((numblocks) * sizeof(int));
	if (q->ledger == NULL) {
		printf("Error: malloc could not allocate q->ledger\n");
		return;
	}

	for (i = 0; i < (numblocks); i++) {
		q->ledger[i] = 0;
	}
		
	q->mut = (pthread_mutex_t *) malloc (sizeof (pthread_mutex_t));
	pthread_mutex_init (q->mut, NULL);
	
	q->produced = (pthread_cond_t *) malloc (sizeof (pthread_cond_t));
	pthread_cond_init (q->produced, NULL);
	q->compressed = (pthread_cond_t *) malloc (sizeof (pthread_cond_t));
	pthread_cond_init (q->compressed, NULL);	
	q->sendready = (pthread_cond_t *) malloc (sizeof (pthread_cond_t));
	pthread_cond_init (q->sendready, NULL);	
	q->sent = (pthread_cond_t *) malloc (sizeof (pthread_cond_t));
	pthread_cond_init (q->sent, NULL);
	
	return (q);
}


void signalExitThreads()
{
	exit_signal++;
	while(! (exit_signal > 3));
}


void queueDelete (queue *q)
{
	int i =0;
	
	signalExitThreads();
	
	pthread_mutex_destroy (q->mut);
	free (q->mut);	
	pthread_cond_destroy (q->produced);
	free (q->produced);
	pthread_cond_destroy (q->compressed);
	free (q->compressed);	
	pthread_cond_destroy (q->sendready);
	free (q->sendready);	
	pthread_cond_destroy (q->sent);
	free (q->sent);

	
	for (i = 0; i < (numblocks); i++) {
		deleteCPUmem(q->bufout[i]);	
		deleteCPUmem(q->buf[i]);
	}
	
	deleteGPUStreams();
	
	free(q->buf);
	free(q->bufout);
	free(q->ledger);	
	free(q->outsize);
	free (q);
	
	
	resetGPU();

}


void  init_compression(queue * fifo,int maxit,int numb,int bsize, char * filename, unsigned int * book)
{
	maxiterations=maxit;
	numblocks=numb;
	blocksize=bsize;
	outputfilename = filename;
	bookkeeping = book;
	printf("Initializing the GPU\n");
	initGPU();
	//create consumer threades
	pthread_create (&congpu, NULL, gpu_consumer, fifo);
	pthread_create (&concpu, NULL, cpu_consumer, fifo);
	pthread_create (&consend, NULL, cpu_sender, fifo);
	
	
	
	return;
}

void join_comp_threads()
{	
	pthread_join (congpu, NULL);
	pthread_join (concpu, NULL);
	pthread_join (consend, NULL);
	exit_signal = 3;
}



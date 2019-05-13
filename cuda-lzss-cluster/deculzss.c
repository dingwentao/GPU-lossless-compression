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
#include "deculzss.h"
#include <sys/time.h>

pthread_t congpu, concpu;

extern unsigned char * decompressed_buffer; 

//int loopnum=0;
int bsize = 0;
int nchunks = 0;
int paddingsize = 0;
char * outfile;

void *degpu_consumer (void *q)
{

	//struct timeval t1_start,t1_end,t2_start,t2_end;
	//double time_d, alltime;

	dequeue *fifo;
	int i, d;
	int success=0;
	fifo = (dequeue *)q;
	int decomp_length=0;
	
	fifo->in_d = deinitGPUmem((int)bsize);
	fifo->out_d = deinitGPUmem((int)bsize*2);
	
	
	for (i = 0; i < nchunks; i++) {
		
		//gettimeofday(&t1_start,0);	
	
		pthread_mutex_lock (fifo->mut);
		while (fifo->ledger[fifo->headGW]!=1) {
			//printf ("gpu consumer: dequeue .\n");
			pthread_cond_wait (fifo->rcvd, fifo->mut);
		}
		pthread_mutex_unlock (fifo->mut);
		
		
		if(fifo->compsize[fifo->headGW] == bsize)
			;//printf("dont do anything %d\n",i);
		else
		{
			success=decompression_kernel_wrapper(fifo->buf[fifo->headGW], fifo->compsize[fifo->headGW]/*recvd size*/, &decomp_length, 0, 1, 1);
			if(!success || decomp_length!=bsize){
				printf("Decompression failed. Success %d\n",success);
			}	
		}
				
		pthread_mutex_lock (fifo->mut);
		fifo->ledger[fifo->headGW]++;
		fifo->headGW++;
		if (fifo->headGW == NUMBUF)
			fifo->headGW = 0;

		pthread_mutex_unlock (fifo->mut);
		
		pthread_cond_signal (fifo->decomp);

		//gettimeofday(&t1_end,0);
		//alltime = (t1_end.tv_sec-t1_start.tv_sec) + (t1_end.tv_usec - t1_start.tv_usec)/1000000.0;
		//printf("GPU whole took:\t%f \n", alltime);
	}
	
	dedeleteGPUmem(fifo->in_d);
	dedeleteGPUmem(fifo->out_d);
	/**/
	return (NULL);
}

void *decpu_consumer (void *q)
{
	FILE *outFile;

	// struct timeval t1_start,t1_end,t2_start,t2_end;
	// double time_d, alltime;

	int i;
	int success=0;
	dequeue *fifo;	
	fifo = (dequeue *)q;
	int comp_length=0;
	
	
	if(outfile!=NULL)
		outFile = fopen(outfile, "wb");
	else
		outFile = fopen("decomp.dat", "wb");
		
	if(outFile == NULL)
	{
		printf ("Output file open error"); exit (2);
	}
	
	for (i = 0; i < nchunks; i++) {

		pthread_mutex_lock (fifo->mut);
		while (fifo->ledger[fifo->headWR]!=2) {
			// printf ("cpu consumer: dequeue EMPTY.\n");
			pthread_cond_wait (fifo->decomp, fifo->mut);
		}
		pthread_mutex_unlock (fifo->mut);

		// WRITE to FILE
		//
		if(i == nchunks-1 && paddingsize>0)
			fwrite(fifo->buf[fifo->headWR],bsize-paddingsize/*comp_length*/, 1, outFile);
		else
			fwrite(fifo->buf[fifo->headWR],bsize/*comp_length*/, 1, outFile);


		pthread_mutex_lock (fifo->mut);
		fifo->ledger[fifo->headWR]=0;
		fifo->headWR++;
		if (fifo->headWR == NUMBUF)
			fifo->headWR = 0;

		pthread_mutex_unlock (fifo->mut);
		
		pthread_cond_signal (fifo->wrote);

	}
	fclose(outFile);
	/**/
	return (NULL);
}

dequeue *dequeueInit (int size, int numchnks,int pad)
{
	bsize = size;
	nchunks = numchnks;
	paddingsize = pad;
	
	dequeue *q;

	q = (dequeue *)malloc (sizeof (dequeue));
	if (q == NULL) return (NULL);

	int i;
	//alloc bufs
	unsigned char ** buffer;
	unsigned char ** bufferout;
	
	for(i=0;i<NUMBUF;i++)
		q->compsize[i]=0;
		
	
	/*  allocate storage for an array of pointers */
	buffer = (unsigned char **)malloc((NUMBUF) * sizeof(unsigned char *));
	if (buffer == NULL) {
		printf("Error: malloc could not allocate buffer\n");
		return;
	}
	bufferout = (unsigned char **)malloc((NUMBUF) * sizeof(unsigned char *));
	if (bufferout == NULL) {
		printf("Error: malloc could not allocate bufferout\n");
		return;
	}
	//DONT HAVE TO ALLOCATE MEM
	//IT IS GIVEN BY THE PROGRAM ANYWAYS
	  
	/* for each pointer, allocate storage for an array of chars */
	for (i = 0; i < (NUMBUF); i++) {
		buffer[i] = (unsigned char *)malloc(bsize * sizeof(unsigned char));
		if (buffer[i] == NULL) {printf ("Memory error, buffer"); exit (2);}
	}
	for (i = 0; i < (NUMBUF); i++) {
		bufferout[i] = (unsigned char *)malloc(bsize * 2 * sizeof(unsigned char));
		if (bufferout[i] == NULL) {printf ("Memory error, bufferout"); exit (2);}
	}
	
	q->buf = buffer;
	q->bufout = bufferout;
	
	q->headRG = 0;
	q->headGW = 0;
	q->headWR = 0;
	
	q->ledger = (int *)malloc((NUMBUF) * sizeof(int));
	if (q->ledger == NULL) {
		printf("Error: malloc could not allocate q->ledger\n");
		return;
	}

	/* for each initialize */
	for (i = 0; i < (NUMBUF); i++) {
		q->ledger[i] = 0;
	}
		
	q->mut = (pthread_mutex_t *) malloc (sizeof (pthread_mutex_t));
	pthread_mutex_init (q->mut, NULL);
	
	q->rcvd = (pthread_cond_t *) malloc (sizeof (pthread_cond_t));
	pthread_cond_init (q->rcvd, NULL);
	q->decomp = (pthread_cond_t *) malloc (sizeof (pthread_cond_t));
	pthread_cond_init (q->decomp, NULL);	
	q->wrote = (pthread_cond_t *) malloc (sizeof (pthread_cond_t));
	pthread_cond_init (q->wrote, NULL);
	
	return (q);
}


void dequeueDelete (dequeue *q)
{
	pthread_mutex_destroy (q->mut);
	free (q->mut);	
	pthread_cond_destroy (q->rcvd);
	free (q->rcvd);
	pthread_cond_destroy (q->decomp);
	free (q->decomp);	
	pthread_cond_destroy (q->wrote);
	free (q->wrote);
	free(q->bufout);	
	free(q->buf);
	free(q->ledger);	
	free (q);

}


void  init_decompression(dequeue * fifo, char * filename)
{
	outfile = filename; 
	printf("Initializing the GPU\n");
	deinitGPU();
	//create consumer threades
	pthread_create (&congpu, NULL, degpu_consumer, fifo);
	pthread_create (&concpu, NULL, decpu_consumer, fifo);
	
	return;
}

void join_decomp_threads()
{
	pthread_join (congpu, NULL);
	pthread_join (concpu, NULL);
}











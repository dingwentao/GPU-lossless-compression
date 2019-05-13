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
#include <string.h>
#include <signal.h>
#include <sys/socket.h>
//#include "decompression.h"

struct timeval tall_start,tall_end;
double alltime_signal;
//int loopcount=0;
dequeue *fifo;	
char * inputfile;
char * outfile;
int bufsize=0;
int numbufs=0;

void *receiver (void *q)
{

	
    FILE *inFile;//, *outFile, *decFile;  /* input & output files */
	inFile = NULL;
	dequeue *fifo;
	int i;	

	fifo = (dequeue *)q;


	if ((inFile = fopen(inputfile, "rb")) == NULL)
	{
		printf ("Memory error, temp"); exit (2);
	}		
	int * readsize;
	readsize = (int *)malloc(sizeof(int)*numbufs);
	
	int size=0;
	
	fread (&size,sizeof(unsigned int), 1,inFile); //trash 
	fread (&size,sizeof(unsigned int), 1,inFile); //trash
	fread (readsize,sizeof(unsigned int), numbufs,inFile);	

	for (i = 0; i < numbufs; i++) {
		

		pthread_mutex_lock (fifo->mut);
		while (fifo->ledger[fifo->headRG]!=0) {
			//printf ("receiver: queue FULL.\n");
			pthread_cond_wait (fifo->wrote, fifo->mut);
		}
		
		size = (readsize[i] - ((i==0)?0:readsize[i-1]));
		
		int result = fread (fifo->buf[fifo->headRG],1,size,inFile);
		if (result != size) {printf ("Reading error1, read %d ",result); exit (3);}	

		fifo->compsize[fifo->headRG]=size;
		
		fifo->ledger[fifo->headRG]++;
		fifo->headRG++;
		if (fifo->headRG == NUMBUF)
			fifo->headRG = 0;

		pthread_mutex_unlock (fifo->mut);
		pthread_cond_signal (fifo->rcvd);
		

	}
	fclose(inFile);
	return (NULL);
}

int decompression(char * filename, int size,char * outfilename) 
{
	// gettimeofday(&tall_start,0);	
	// double alltime;
	
	int padding=0;  	
	inputfile = filename;  
	outfile = outfilename;  
	bufsize = size;

    FILE *inFile;//, *outFile, *decFile;  /* input & output files */
	if ((inFile = fopen(inputfile, "rb")) == NULL)
	{
		printf ("Memory error, temp"); exit (2);
	}
	
	fread (&numbufs,sizeof(unsigned int), 1,inFile);
	fread (&padding,sizeof(unsigned int), 1,inFile);
	fclose(inFile);
	
	printf("Num bufs %d padding size %d bufsize %d\n", numbufs,padding,bufsize);
	
	pthread_t rce;
	
	fifo = dequeueInit (bufsize,numbufs,padding);
	if (fifo ==  NULL) {
		fprintf (stderr, "main: Queue Init failed.\n");
		exit (1);
	}		

	//init compression threads
	init_decompression(fifo,outfile);
	
	//create receiver
	pthread_create (&rce, NULL, receiver, fifo);
	//start producing
	//and start signaling as we go 

	//join all 
	join_decomp_threads();
	//join receiver
	pthread_join (rce, NULL);
	
	dequeueDelete (fifo);

	
	//exit
	return 0;

} 





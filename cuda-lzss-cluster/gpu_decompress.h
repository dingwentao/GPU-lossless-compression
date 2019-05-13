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

#ifndef __GPU_DECOMPRESS_H_
#define __GPU_DECOMPRESS_H_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/***************************************************************************
*                                CONSTANTS
***************************************************************************/
#define FALSE   0
#define TRUE    1

#define WINDOW_SIZE 128

/* maximum match length not encoded and encoded (4 bits) */
#define MAX_UNCODED     2
#define MAX_CODED       128
#define NWORKERS	2

#define PCKTSIZE 4096

/***************************************************************************
*                            TYPE DEFINITIONS
***************************************************************************/
/* unpacked encoded offset and length, gets packed into 12 bits and 4 bits*/
typedef struct deencoded_string_t
{
    int offset;     /* offset to start of longest match */
    int length;     /* length of longest match */
} deencoded_string_t;

typedef enum
{
    ENCODE,
    DECODE
} MODES;

struct dethread_data{
   int  tid;
   int  numthreads;
   unsigned char *buf;
   unsigned char *outbuf;
   int * header;
   int msg_length;
   int success;
   unsigned char * in_d;
   int ** ledger;
   float timings;
   int nstreams;
   int * sizes;
};




/***************************************************************************
*                                FUNCTIONS
***************************************************************************/

extern "C" int  decompression_kernel_wrapper(unsigned char *buffer, int buf_length, int * comp_length, int compression_type, int wsize, int numthre);
extern "C" unsigned char * deinitGPUmem( int buf_length);
extern "C" void dedeleteGPUmem(unsigned char * mem_d);
extern "C" void deinitGPU();
#endif


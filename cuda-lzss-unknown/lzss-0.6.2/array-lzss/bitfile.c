/***************************************************************************
 *                        Bit Stream File Implementation
 *
 *   File    : bitfile.c
 *   Purpose : This file implements a simple library of I/O functions for
 *             files that contain data in sizes that aren't integral bytes.
 *             An attempt was made to make the functions in this library
 *             analogous to functions provided to manipulate byte streams.
 *             The functions contained in this library were created with
 *             compression algorithms in mind, but may be suited to other
 *             applications.
 *   Author  : Michael Dipperstein
 *   Date    : January 9, 2004
 *
 ****************************************************************************
 *   UPDATES
 *
 *   $Id: bitfile.c,v 1.13 2008/09/15 04:10:20 michael Exp $
 *   $Log: bitfile.c,v $
 *   Revision 1.13  2008/09/15 04:10:20  michael
 *   Removed dead code.
 *
 *   Revision 1.12  2008/01/25 07:03:49  michael
 *   Added BitFileFlushOutput().
 *
 *   Revision 1.11  2007/12/30 23:55:30  michael
 *   Corrected errors in BitFileOpen and MakeBitFile reported by an anonymous
 *   user.  Segment faults may have occurred if fopen returned a NULL.
 *
 *   Revision 1.10  2007/08/26 21:53:48  michael
 *   Changes required for LGPL v3.
 *
 *   Revision 1.9  2007/07/10 05:34:07  michael
 *   Remove ',' after last element in the enum endian_t.
 *
 *   Revision 1.8  2007/02/06 06:22:07  michael
 *   Used trim program to remove trailing spaces.
 *
 *   Revision 1.7  2006/06/03 19:32:38  michael
 *   Corrected error reporetd anonymous.  The allocation of constants used to
 *   open underlying read/write/append files did not account for a terminating
 *   null.
 *
 *   Used spell checker to correct spelling.
 *
 *   Revision 1.6  2005/12/08 06:56:55  michael
 *   Minor text corrections.
 *
 *   Revision 1.5  2005/12/06 15:06:37  michael
 *   Added BitFileGetBitsInt and BitFilePutBitsInt for integer types.
 *
 *   Revision 1.4  2005/06/23 04:34:18  michael
 *   Prevent BitfileGetBits/PutBits from accessing an extra byte when given
 *   an integral number of bytes.
 *
 *   Revision 1.3  2004/11/09 14:16:58  michael
 *   Added functions to convert open bit_file_t to FILE and to
 *   align open bit_file_t to the next byte.
 *
 *   Revision 1.2  2004/06/15 13:15:58  michael
 *   Use incomplete type to hide definition of bitfile structure
 *
 *   Revision 1.1.1.1  2004/02/09 05:31:42  michael
 *   Initial release
 *
 *
 ****************************************************************************
 *
 * Bitfile: Bit stream File I/O Routines
 * Copyright (C) 2004-2007 by Michael Dipperstein (mdipper@cs.ucsb.edu)
 *
* This file is part of the bit file library.
*
* The bit file library is free software; you can redistribute it and/or
* modify it under the terms of the GNU Lesser General Public License as
* published by the Free Software Foundation; either version 3 of the
* License, or (at your option) any later version.
*
* The bit file library is distributed in the hope that it will be useful,
    * but WITHOUT ANY WARRANTY; without even the implied warranty of
    * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser
    * General Public License for more details.
    *
    * You should have received a copy of the GNU Lesser General Public License
    * along with this program.  If not, see <http://www.gnu.org/licenses/>.
    *
    ***************************************************************************/

    /***************************************************************************
     *                             INCLUDED FILES
     ***************************************************************************/
#include <stdlib.h>
#include <errno.h>
#include "bitfile.h"

    /***************************************************************************
     *                            TYPE DEFINITIONS
     ***************************************************************************/

    typedef enum
{
    BF_UNKNOWN_ENDIAN,
    BF_LITTLE_ENDIAN,
    BF_BIG_ENDIAN
} endian_t;

struct bit_file_t
{
    unsigned char *outBuffer;
    unsigned int  outBytes;
    unsigned char bitBuffer;    /* bits waiting to be read/written */
    unsigned char bitCount;     /* number of bits in bitBuffer */
    endian_t endian; 
};

/* union used to test for endianess */
typedef union
{
    unsigned long word;
    unsigned char bytes[sizeof(unsigned long)];
} endian_test_t;




/* Own output buffer definitions */
#define LENGTH 256
static unsigned char outputBuffer[LENGTH];
static unsigned int outputBytes = 0;
static const char *outFile = "/home/vm18648/Desktop/myout.txt";

/* Routine for printing buffer */
void printBuffer();
void putInBuffer(unsigned char c);
void putInBufferStream(unsigned char c,bit_file_t * stream);
void printBufferStream(bit_file_t * stream);

void FreeBitStream(bit_file_t * stream);
bit_file_t *BitStreamOpen(unsigned char *buffer);

/***************************************************************************
 *                               PROTOTYPES
 ***************************************************************************/
endian_t DetermineEndianess(void);

int BitFilePutBitsLE(bit_file_t *stream, void *bits, const unsigned int count);
int BitFilePutBitsBE(bit_file_t *stream, void *bits, const unsigned int count,
	const size_t size);


/***************************************************************************
 *                                FUNCTIONS
 ***************************************************************************/

/* Opening BitStream */
bit_file_t *BitStreamOpen(unsigned char *buffer)
{
    bit_file_t *bf;

    bf = (bit_file_t *)malloc(sizeof(bit_file_t));

    if (bf == NULL)
    {
	/* malloc failed */
	errno = ENOMEM;
    }
    else
    {       bf->outBuffer = buffer;
	bf->outBytes = 0;
	bf->bitBuffer = 0;
	bf->bitCount = 0;
	bf->endian = DetermineEndianess();
    }

    return (bf);
}


/***************************************************************************
 *   Function   : DetermineEndianess
 *   Description: This function determines the endianess of the current
 *                hardware architecture.  An unsigned long is set to 1.  If
 *                the 1st byte of the unsigned long gets the 1, this is a
 *                little endian machine.  If the last byte gets the 1, this
 *                is a big endian machine.
 *   Parameters : None
 *   Effects    : None
 *   Returned   : endian_t for current machine architecture
 ***************************************************************************/
endian_t DetermineEndianess(void)
{
    endian_t endian;
    endian_test_t endianTest;

    endianTest.word = 1;

    if (endianTest.bytes[0] == 1)
    {
	/* LSB is 1st byte (little endian)*/
	endian = BF_LITTLE_ENDIAN;
    }
    else if (endianTest.bytes[sizeof(unsigned long) - 1] == 1)
    {
	/* LSB is last byte (big endian)*/
	endian = BF_BIG_ENDIAN;
    }
    else
    {
	endian = BF_UNKNOWN_ENDIAN;
    }

    return endian;
}

/***************************************************************************
 *   Function   : BitFileByteAlign
 *   Description: This function aligns the bitfile to the nearest byte.  For
 *                output files, this means writing out the bit buffer with
 *                extra bits set to 0.  For input files, this means flushing
 *                the bit buffer.
 *   Parameters : stream - pointer to bit file stream to align
 *   Effects    : Flushes out the bit buffer.
 *   Returned   : EOF if stream is NULL or write fails.  Writes return the
 *                byte aligned contents of the bit buffer.  Reads returns
 *                the unaligned contents of the bit buffer.
 ***************************************************************************/
/*int BitFileByteAlign(bit_file_t *stream)
  {
  int returnValue;

  if (stream == NULL)
  {
  return(EOF);
  }

  returnValue = stream->bitBuffer;

  if ((stream->mode == BF_WRITE) || (stream->mode == BF_APPEND))
  {  */
/* write out any unwritten bits */
/*       if (stream->bitCount != 0)
	 {
	 (stream->bitBuffer) <<= 8 - (stream->bitCount);
	 fputc(stream->bitBuffer, stream->fp);  */ /* handle error? */
/*        }
	  }

	  stream->bitBuffer = 0;
	  stream->bitCount = 0;

	  return (returnValue);
	  } */

/***************************************************************************
 *   Function   : BitFilePutChar
 *   Description: This function writes the byte passed as a parameter to the
 *                file passed a parameter.
 *   Parameters : c - the character to be written
 *                stream - pointer to bit file stream to write to
 *   Effects    : Writes a byte to the file and updates buffer accordingly.
 *   Returned   : On success, the character written, otherwise EOF.
 ***************************************************************************/
int BitFilePutChar(const int c, bit_file_t *stream)
{
    unsigned char tmp;

    if (stream == NULL)
    {
	return(EOF);
    }

    if (stream->bitCount == 0)
    {

	/* Printing to buffer */
	putInBuffer(c);
	putInBufferStream(c,stream);
	return c;
    }

    /* figure out what to write */
    tmp = ((unsigned char)c) >> (stream->bitCount);
    tmp = tmp | ((stream->bitBuffer) << (8 - stream->bitCount));


    /* Printing to buffer */
    putInBuffer(tmp);
    putInBufferStream(tmp,stream);
    /* We shud add stream->bitBuffer = c here */
    stream->bitBuffer = c;  /* VERY CAREFUL */

    return tmp;
}


/***************************************************************************
 *   Function   : BitFilePutBit
 *   Description: This function writes the bit passed as a parameter to the
 *                file passed a parameter.
 *   Parameters : c - the bit value to be written
 *                stream - pointer to bit file stream to write to
 *   Effects    : Writes a bit to the bit buffer.  If the buffer has a byte,
 *                the buffer is written to the file and cleared.
 *   Returned   : On success, the bit value written, otherwise EOF.
 ***************************************************************************/
int BitFilePutBit(const int c, bit_file_t *stream)
{
    int returnValue = c;

    if (stream == NULL)
    {
	return(EOF);
    }

    stream->bitCount++;
    stream->bitBuffer <<= 1;

    if (c != 0)
    {
	stream->bitBuffer |= 1;
    }

    /* write bit buffer if we have 8 bits */
    if (stream->bitCount == 8)
    {
	/* Printing in buffer */
	putInBuffer(stream->bitBuffer); 
	putInBufferStream(stream->bitBuffer,stream);

	/* reset buffer */
	stream->bitCount = 0;
	stream->bitBuffer = 0;
    }

    return returnValue;
}

/***************************************************************************
 *   Function   : BitFilePutBits
 *   Description: This function writes the specified number of bits from the
 *                memory location passed as a parameter to the file passed
 *                as a parameter.   Bits are written msb to lsb.
 *   Parameters : stream - pointer to bit file stream to write to
 *                bits - pointer to bits to write
 *                count - number of bits to write
 *   Effects    : Writes bits to the bit buffer and file stream.  The bit
 *                buffer will be modified as necessary.
 *   Returned   : EOF for failure, otherwise the number of bits written.  If
 *                an error occurs after a partial write, the partially
 *                written bits will not be unwritten.
 ***************************************************************************/
int BitFilePutBits(bit_file_t *stream, void *bits, const unsigned int count)
{
    unsigned char *bytes, tmp;
    int offset, remaining, returnValue;

    bytes = (unsigned char *)bits;

    if ((stream == NULL) || (bits == NULL))
    {
	return(EOF);
    }

    offset = 0;
    remaining = count;

    /* write whole bytes */
    while (remaining >= 8)
    {
	returnValue = BitFilePutChar(bytes[offset], stream);

	if (returnValue == EOF)
	{
	    return EOF;
	}

	remaining -= 8;
	offset++;
    }

    if (remaining != 0)
    {
	/* write remaining bits */
	tmp = bytes[offset];

	while (remaining > 0)
	{
	    returnValue = BitFilePutBit((tmp & 0x80), stream);

	    if (returnValue == EOF)
	    {
		return EOF;
	    }

	    tmp <<= 1;
	    remaining--;
	}
    }

    return count;
}

/***************************************************************************
 *   Function   : BitFilePutBitsInt
 *   Description: This function provides a machine independent layer that
 *                allows a single function call to write an arbitrary number
 *                of bits from an integer type variable into a file.
 *   Parameters : stream - pointer to bit file stream to write to
 *                bits - pointer to bits to write
 *                count - number of bits to write
 *                size - sizeof type containing "bits"
 *   Effects    : Calls a function that writes bits to the bit buffer and
 *                file stream.  The bit buffer will be modified as necessary.
 *                the bits will be written to the file stream from least
 *                significant byte to most significant byte.
 *   Returned   : EOF for failure, otherwise the number of bits written.  If
 *                an error occurs after a partial write, the partially
 *                written bits will not be unwritten.
 ***************************************************************************/
int BitFilePutBitsInt(bit_file_t *stream, void *bits, const unsigned int count,
	const size_t size)
{
    int returnValue;

    if ((stream == NULL) || (bits == NULL))
    {
	return(EOF);
    }

    if (stream->endian == BF_LITTLE_ENDIAN)
    {
	returnValue = BitFilePutBitsLE(stream, bits, count);
    }
    else if (stream->endian == BF_BIG_ENDIAN)
    {
	returnValue = BitFilePutBitsBE(stream, bits, count, size);
    }
    else
    {
	returnValue = EOF;
    }

    return returnValue;
}

/***************************************************************************
 *   Function   : BitFilePutBitsLE   (Little Endian)
 *   Description: This function writes the specified number of bits from the
 *                memory location passed as a parameter to the file passed
 *                as a parameter.   Bits are written LSB to MSB.
 *   Parameters : stream - pointer to bit file stream to write to
 *                bits - pointer to bits to write
 *                count - number of bits to write
 *   Effects    : Writes bits to the bit buffer and file stream.  The bit
 *                buffer will be modified as necessary.  bits is treated as
 *                a little endian integer of length >= (count/8) + 1.
 *   Returned   : EOF for failure, otherwise the number of bits written.  If
 *                an error occurs after a partial write, the partially
 *                written bits will not be unwritten.
 ***************************************************************************/
int BitFilePutBitsLE(bit_file_t *stream, void *bits, const unsigned int count)
{
    unsigned char *bytes, tmp;
    int offset, remaining, returnValue;

    bytes = (unsigned char *)bits;
    offset = 0;
    remaining = count;

    /* write whole bytes */
    while (remaining >= 8)
    {
	returnValue = BitFilePutChar(bytes[offset], stream);

	if (returnValue == EOF)
	{
	    return EOF;
	}

	remaining -= 8;
	offset++;
    }

    if (remaining != 0)
    {
	/* write remaining bits */
	tmp = bytes[offset];
	tmp <<= (8 - remaining);

	while (remaining > 0)
	{
	    returnValue = BitFilePutBit((tmp & 0x80), stream);

	    if (returnValue == EOF)
	    {
		return EOF;
	    }

	    tmp <<= 1;
	    remaining--;
	}
    }

    return count;
}

/***************************************************************************
 *   Function   : BitFilePutBitsBE   (Big Endian)
 *   Description: This function writes the specified number of bits from the
 *                memory location passed as a parameter to the file passed
 *                as a parameter.   Bits are written LSB to MSB.
 *   Parameters : stream - pointer to bit file stream to write to
 *                bits - pointer to bits to write
 *                count - number of bits to write
 *   Effects    : Writes bits to the bit buffer and file stream.  The bit
 *                buffer will be modified as necessary.  bits is treated as
 *                a big endian integer of length size.
 *   Returned   : EOF for failure, otherwise the number of bits written.  If
 *                an error occurs after a partial write, the partially
 *                written bits will not be unwritten.
 ***************************************************************************/
int BitFilePutBitsBE(bit_file_t *stream, void *bits, const unsigned int count,
	const size_t size)
{
    unsigned char *bytes, tmp;
    int offset, remaining, returnValue;

    if (count > (size * 8))
    {
	/* too many bits to write */
	return EOF;
    }

    bytes = (unsigned char *)bits;
    offset = size - 1;
    remaining = count;

    /* write whole bytes */
    while (remaining >= 8)
    {
	returnValue = BitFilePutChar(bytes[offset], stream);

	if (returnValue == EOF)
	{
	    return EOF;
	}

	remaining -= 8;
	offset--;
    }

    if (remaining != 0)
    {
	/* write remaining bits */
	tmp = bytes[offset];
	tmp <<= (8 - remaining);

	while (remaining > 0)
	{
	    returnValue = BitFilePutBit((tmp & 0x80), stream);

	    if (returnValue == EOF)
	    {
		return EOF;
	    }

	    tmp <<= 1;
	    remaining--;
	}
    }

    return count;
}

/* Freeing Bit Stream */
void FreeBitStream(bit_file_t *stream){

    free(stream);

}

/* Printing our output buffer */
void putInBuffer(unsigned char c) {

    outputBuffer[outputBytes] = c;
    outputBytes++;

}

/* Printing our output buffer */
void putInBufferStream(unsigned char c,bit_file_t * stream) {

    (stream->outBuffer)[stream->outBytes] = c;
    stream->outBytes++;

}

/* Printing our output buffer */
void printBufferStream(bit_file_t * stream) {
    int i;
    FILE *fp = NULL;
    if((fp = fopen(outFile,"wb")) == NULL) {
	printf("Couldn't open file. Will not print buffer\n");
    }
    printf("Printing Buffer\n");
    for ( i = 0 ; i < stream->outBytes ; i++){
	fputc(stream->outBuffer[i],fp);
	printf("%c",stream->outBuffer[i]);
    }

    fclose(fp);
}


/* Printing our output buffer */
void printBuffer() {
    int i;
    FILE *fp = NULL;
    if((fp = fopen(outFile,"wb")) == NULL) {
	printf("Couldn't open file. Will not print buffer\n");
    }
    printf("Printing Buffer\n");
    for ( i = 0 ; i < outputBytes ; i++){
	/* fprintf(fp,"%c",outputBuffer[i]); */
	fputc(outputBuffer[i],fp);
	printf("%c",outputBuffer[i]);
    }

    fclose(fp);
}

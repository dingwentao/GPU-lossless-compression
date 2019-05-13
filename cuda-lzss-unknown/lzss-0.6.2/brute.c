/***************************************************************************
*          Lempel, Ziv, Storer, and Szymanski Encoding and Decoding
*
*   File    : brute.c
*   Purpose : Implement brute force matching of uncoded strings for LZSS
*             algorithm.
*   Author  : Michael Dipperstein
*   Date    : February 18, 2004
*
****************************************************************************
*   UPDATES
*
*   $Id: brute.c,v 1.4 2007/09/20 04:34:25 michael Exp $
*   $Log: brute.c,v $
*   Revision 1.4  2007/09/20 04:34:25  michael
*   Changes required for LGPL v3.
*
*   Revision 1.3  2006/12/26 04:09:09  michael
*   Updated e-mail address and minor text clean-up.
*
*   Revision 1.2  2005/12/28 06:03:30  michael
*   Use slower but clearer Get/PutBitsInt for reading/writing bits.
*   Replace mod with conditional Wrap macro.
*
*   Revision 1.1  2004/02/22 17:21:37  michael
*   Initial revision of brute force search.  Mostly code removed from lzss.c.
*
*
****************************************************************************
*
* Brute: Brute force matching routines used by LZSS Encoding/Decoding
*        Routine
* Copyright (C) 2004-2007 by
* Michael Dipperstein (mdipper@alumni.engr.ucsb.edu)
*
* This file is part of the lzss library.
*
* The lzss library is free software; you can redistribute it and/or
* modify it under the terms of the GNU Lesser General Public License as
* published by the Free Software Foundation; either version 3 of the
* License, or (at your option) any later version.
*
* The lzss library is distributed in the hope that it will be useful, but
* WITHOUT ANY WARRANTY; without even the implied warranty of
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
#include "lzlocal.h"

/***************************************************************************
*                            GLOBAL VARIABLES
***************************************************************************/
/* cyclic buffer sliding window of already read characters */
extern unsigned char slidingWindow[];
extern unsigned char uncodedLookahead[];

/****************************************************************************
*   Function   : InitializeSearchStructures
*   Description: This function initializes structures used to speed up the
*                process of mathcing uncoded strings to strings in the
*                sliding window.  The brute force search doesn't use any
*                special structures, so this function doesn't do anything.
*   Parameters : None
*   Effects    : None
*   Returned   : None
****************************************************************************/
void InitializeSearchStructures()
{
    return;
}

/****************************************************************************
*   Function   : FindMatch
*   Description: This function will search through the slidingWindow
*                dictionary for the longest sequence matching the MAX_CODED
*                long string stored in uncodedLookahed.
*   Parameters : windowHead - head of sliding window
*                uncodedHead - head of uncoded lookahead buffer
*   Effects    : None
*   Returned   : The sliding window index where the match starts and the
*                length of the match.  If there is no match a length of
*                zero will be returned.
****************************************************************************/
encoded_string_t FindMatch(unsigned int windowHead, unsigned int uncodedHead)
{
    encoded_string_t matchData;
    unsigned int i, j;

    matchData.length = 0;
    matchData.offset = 0;
    i = windowHead;  /* start at the beginning of the sliding window */
    j = 0;

    while (TRUE)
    {
        if (slidingWindow[i] == uncodedLookahead[uncodedHead])
        {
            /* we matched one how many more match? */
            j = 1;

            while(slidingWindow[Wrap((i + j), WINDOW_SIZE)] ==
                uncodedLookahead[Wrap((uncodedHead + j), MAX_CODED)])
            {
                if (j >= MAX_CODED)
                {
                    break;
                }
                j++;
            }

            if (j > matchData.length)
            {
                matchData.length = j;
                matchData.offset = i;
            }
        }

        if (j >= MAX_CODED)
        {
            matchData.length = MAX_CODED;
            break;
        }

        i = Wrap((i + 1), WINDOW_SIZE);
        if (i == windowHead)
        {
            /* we wrapped around */
            break;
        }
    }

    return matchData;
}

/****************************************************************************
*   Function   : ReplaceChar
*   Description: This function replaces the character stored in
*                slidingWindow[charIndex] with the one specified by
*                replacement.
*   Parameters : charIndex - sliding window index of the character to be
*                            removed from the linked list.
*   Effects    : slidingWindow[charIndex] is replaced by replacement.
*   Returned   : None
****************************************************************************/
void ReplaceChar(unsigned int charIndex, unsigned char replacement)
{
    slidingWindow[charIndex] = replacement;
}

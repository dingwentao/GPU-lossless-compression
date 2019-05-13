/***************************************************************************
*          Lempel, Ziv, Storer, and Szymanski Encoding and Decoding
*
*   File    : list.c
*   Purpose : Implement linked list optimized matching of uncoded strings
*             for LZSS algorithm.
*   Author  : Michael Dipperstein
*   Date    : February 18, 2004
*
****************************************************************************
*   UPDATES
*
*   $Id: list.c,v 1.4 2007/09/20 04:34:25 michael Exp $
*   $Log: list.c,v $
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
*   Revision 1.1  2004/02/22 17:24:02  michael
*   Initial revision of linked list search.  Mostly code from lzlist.c.
*
*
****************************************************************************
*
* List: Linked list optimized matching routines used by LZSS
*       Encoding/Decoding Routine
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
#include <limits.h>
#include "lzlocal.h"

/***************************************************************************
*                                CONSTANTS
***************************************************************************/
#define NULL_INDEX      (WINDOW_SIZE + 1)

/***************************************************************************
*                            GLOBAL VARIABLES
***************************************************************************/
/* cyclic buffer sliding window of already read characters */
extern unsigned char slidingWindow[];
extern unsigned char uncodedLookahead[];

unsigned int lists[UCHAR_MAX];                    /* heads of linked lists */
unsigned int next[WINDOW_SIZE];             /* indices of next in list */

/****************************************************************************
*   Function   : InitializeSearchStructures
*   Description: This function initializes structures used to speed up the
*                process of mathcing uncoded strings to strings in the
*                sliding window.  For link list optimized searches, this
*                means that linked lists of strings all starting with
*                the same character are initialized.
*   Parameters : None
*   Effects    : Initializes lists and next array
*   Returned   : None
*
*   NOTE: This function assumes that the sliding window is initially filled
*         with all identical characters.
****************************************************************************/
void InitializeSearchStructures()
{
    unsigned int i;

    for (i = 0; i < WINDOW_SIZE; i++)
    {
        next[i] = i + 1;
    }

    /* there's no next for the last character */
    next[WINDOW_SIZE - 1] = NULL_INDEX;

    /* the only list right now is the ' ' list */
    for (i = 0; i < 256; i++)
    {
        lists[i] = NULL_INDEX;
    }

    lists[' '] = 0;
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
    i = lists[uncodedLookahead[uncodedHead]];   /* start of proper list */

    while (i != NULL_INDEX)
    {
        /* the list insures we matched one, how many more match? */
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

        if (j >= MAX_CODED)
        {
            matchData.length = MAX_CODED;
            break;
        }

        i = next[i];    /* try next in list */
    }

    return matchData;
}

/****************************************************************************
*   Function   : AddChar
*   Description: This function adds the character stored in
*                slidingWindow[charIndex] to the linked lists.
*   Parameters : charIndex - sliding window index of the character to be
*                            added to the linked list.
*   Effects    : slidingWindow[charIndex] appended to the end of the
*                appropriate linked list.
*   Returned   : NONE
****************************************************************************/
void AddChar(unsigned int charIndex)
{
    unsigned int i;

    /* inserted character will be at the end of the list */
    next[charIndex] = NULL_INDEX;

    if (lists[slidingWindow[charIndex]] == NULL_INDEX)
    {
        /* this is the only character in it's list */
        lists[slidingWindow[charIndex]] = charIndex;
        return;
    }

    /* find the end of the list */
    i = lists[slidingWindow[charIndex]];
    while(next[i] != NULL_INDEX)
    {
        i = next[i];
    }

    /* add new character to the list end */
    next[i] = charIndex;
}

/****************************************************************************
*   Function   : RemoveChar
*   Description: This function removes the character stored in
*                slidingWindow[charIndex] from the linked lists.
*   Parameters : charIndex - sliding window index of the character to be
*                            removed from the linked list.
*   Effects    : slidingWindow[charIndex] is removed from it's linked list
*                and the list is appropriately reconnected.
*   Returned   : NONE
****************************************************************************/
void RemoveChar(unsigned int charIndex)
{
    unsigned int i;
    unsigned int nextIndex;

    nextIndex = next[charIndex];        /* remember where this points to */
    next[charIndex] = NULL_INDEX;

    if (lists[slidingWindow[charIndex]] == charIndex)
    {
        /* we're deleting a list head */
        lists[slidingWindow[charIndex]] = nextIndex;
        return;
    }

    /* find character pointing to ours */
    i = lists[slidingWindow[charIndex]];

    while(next[i] != charIndex)
    {
        i = next[i];
    }

    /* point the previous next */
    next[i] = nextIndex;
}

/****************************************************************************
*   Function   : ReplaceChar
*   Description: This function replaces the character stored in
*                slidingWindow[charIndex] with the one specified by
*                replacement.  The linked list entries effected by the
*                replacement are also corrected.
*   Parameters : charIndex - sliding window index of the character to be
*                            removed from the linked list.
*   Effects    : slidingWindow[charIndex] is replaced by replacement.  Old
*                list entries for strings containing slidingWindow[charIndex]
*                are removed and new ones are added.
*   Returned   : None
****************************************************************************/
void ReplaceChar(unsigned int charIndex, unsigned char replacement)
{
    RemoveChar(charIndex);
    slidingWindow[charIndex] = replacement;
    AddChar(charIndex);
}

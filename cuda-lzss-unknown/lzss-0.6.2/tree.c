/***************************************************************************
*          Lempel, Ziv, Storer, and Szymanski Encoding and Decoding
*
*   File    : tree.c
*   Purpose : Implement a binary tree used by the LZSS algorithm to search
*             for matching of uncoded strings in the sliding window.
*   Author  : Michael Dipperstein
*   Date    : August 16, 2010
*
****************************************************************************
*
* Tree: Tree table optimized matching routines used by LZSS
*       Encoding/Decoding Routine
* Copyright (C) 2010 by
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
#include <stdio.h>
#include <ctype.h>
#include "lzlocal.h"

/***************************************************************************
*                                CONSTANTS
***************************************************************************/
#define ROOT_INDEX      (WINDOW_SIZE + 1)
#define NULL_INDEX      (ROOT_INDEX + 1)

/***************************************************************************
*                            TYPE DEFINITIONS
***************************************************************************/

/***************************************************************************
* This data structure represents the node of a binary search tree.  The
* leftChild, rightChild, and parent fields should contain the index of the
* slidingWindow dictionary where the left child, right child, and parent
* strings begin.
***************************************************************************/
typedef struct tree_node_t
{
    unsigned int leftChild;
    unsigned int rightChild;
    unsigned int parent;
} tree_node_t;

/***************************************************************************
*                                 MACROS
***************************************************************************/
/* reattach children after a node insertion */
#define FixChildren(idx)	if (tree[(idx)].leftChild != NULL_INDEX)\
								tree[tree[(idx)].leftChild].parent = (idx);\
							if (tree[(idx)].rightChild != NULL_INDEX)\
								tree[tree[(idx)].rightChild].parent = (idx);


/***************************************************************************
*                            GLOBAL VARIABLES
***************************************************************************/
/* cyclic buffer sliding window of already read characters */
extern unsigned char slidingWindow[WINDOW_SIZE];
extern unsigned char uncodedLookahead[MAX_CODED];

tree_node_t tree[WINDOW_SIZE];		/* tree[n] is node for slidingWindow[n] */
unsigned int treeRoot;				/* index of the root of the tree */

/***************************************************************************
*                               PROTOTYPES
***************************************************************************/
/* add/remove strings starting at slidingWindow[charIndex] too/from tree */
void AddString(unsigned int charIndex);
void RemoveString(unsigned int charIndex);

/* debugging functions not used by algorithm */
void PrintLen(unsigned int charIndex, unsigned int len);
void DumpTree(unsigned int root);

/****************************************************************************
*   Function   : InitializeSearchStructures
*   Description: This function initializes structures used to speed up the
*                process of matching uncoded strings to strings in the
*                sliding window.  For binary searches, this means that the
*                last MAX_CODED length string in the sliding window will be
* 				 made the root of the tree, and have no children.  This only
*                works if the sliding window is filled with identical
*                symbols.
*   Parameters : None
*   Effects    : A tree consisting of just a root node is created.
*   Returned   : None
*
*   NOTE: This function assumes that the sliding window is initially filled
*         with all identical characters.
****************************************************************************/
void InitializeSearchStructures()
{
    unsigned int i;

	/* clear out all tree node pointers */
    for (i = 0; i < WINDOW_SIZE; i++)
    {
        tree[i].leftChild = NULL_INDEX;
        tree[i].rightChild = NULL_INDEX;
        tree[i].parent = NULL_INDEX;
    }

    /************************************************************************
    * Since the encode routine only fills the sliding window with one
    * character, there is only possible MAX_CODED length string in the tree.
    * Use the newest of those strings at the tree root.
    ************************************************************************/
    treeRoot = (WINDOW_SIZE - MAX_CODED) - 1;
    tree[treeRoot].parent = ROOT_INDEX;
}

/****************************************************************************
*   Function   : FindMatch
*   Description: This function will search through the slidingWindow
*                dictionary for the longest sequence matching the MAX_CODED
*                long string stored in uncodedLookahead.
*   Parameters : windowHead - not used
*                uncodedHead - head of uncoded lookahead buffer
*   Effects    : NONE
*   Returned   : The sliding window index where the match starts and the
*                length of the match.  If there is no match a length of
*                zero will be returned.
****************************************************************************/
encoded_string_t FindMatch(unsigned int windowHead, unsigned int uncodedHead)
{
    encoded_string_t matchData;
    unsigned int i, j;
    int compare;

    matchData.length = 0;
    matchData.offset = 0;

    i = treeRoot;       /* start at root */
    j = 0;

    while (i != NULL_INDEX)
    {
        compare = slidingWindow[i] - uncodedLookahead[uncodedHead];

        if (0 == compare)
        {
            /* we matched the first symbol, how many more match? */
            j = 1;

            while((compare = slidingWindow[Wrap((i + j), WINDOW_SIZE)] -
                uncodedLookahead[Wrap((uncodedHead + j), MAX_CODED)]) == 0)
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
			/* we found the largest allowed match */
            matchData.length = MAX_CODED;
            break;
        }

        if (compare > 0)
        {
            /* branch left and look for a longer match */
            i = tree[i].leftChild;
        }
        else
        {
            /* branch right and look for a longer match */
            i = tree[i].rightChild;
        }
    }

    return matchData;
}

/****************************************************************************
*   Function   : CompareString
*   Description: This function will compare two MAX_CODED long strings in
*                the slidingWindow dictionary.
*   Parameters : index1 - slidingWindow index where the first string starts
*                index2 - slidingWindow index where the second string starts
*   Effects    : NONE
*   Returned   : 0 if first string equals second string.
*                < 0 if first string is less than second string.
*                > 0 if first string is greater than second string.
****************************************************************************/
int CompareString(unsigned int index1, unsigned int index2)
{
    unsigned int offset;
    int result = 0;

    for (offset = 0; offset < MAX_CODED; offset++)
    {
        result = slidingWindow[Wrap((index1 + offset), WINDOW_SIZE)] -
            slidingWindow[Wrap((index2 + offset), WINDOW_SIZE)];

        if (result != 0)
        {
            break;      /* we have a mismatch */
        }
    }

    return result;
}

/****************************************************************************
*   Function   : AddString
*   Description: This function adds the MAX_UNCODED long string starting at
*                slidingWindow[charIndex] to the binary tree.
*   Parameters : charIndex - sliding window index of the string to be
*                            added to the binary tree list.
*   Effects    : The string starting at slidingWindow[charIndex] is inserted
*                into the sorted binary tree.
*   Returned   : NONE
****************************************************************************/
void AddString(unsigned int charIndex)
{
    int compare;
    unsigned int here;

    compare = CompareString(charIndex, treeRoot);

    if (0 == compare)
    {
        /* make start the new root, because it's newer */
        tree[charIndex].leftChild = tree[treeRoot].leftChild;
        tree[charIndex].rightChild = tree[treeRoot].rightChild;
        tree[charIndex].parent = ROOT_INDEX;
        FixChildren(charIndex);

        /* remove old root from the tree */
        tree[treeRoot].leftChild = NULL_INDEX;
        tree[treeRoot].rightChild = NULL_INDEX;
        tree[treeRoot].parent = NULL_INDEX;

        treeRoot = charIndex;
        return;
    }

    here = treeRoot;

    while(TRUE)
    {
        if (compare < 0)
        {
            /* branch left for < */
            if (tree[here].leftChild != NULL_INDEX)
            {
                here = tree[here].leftChild;
            }
            else
            {
                /* we've hit the bottom */
                tree[here].leftChild = charIndex;
                tree[charIndex].leftChild = NULL_INDEX;
                tree[charIndex].rightChild = NULL_INDEX;
                tree[charIndex].parent = here;
                FixChildren(charIndex);
                return;
            }
        }
        else if (compare > 0)
        {
            /* branch right for > */
            if (tree[here].rightChild != NULL_INDEX)
            {
                here = tree[here].rightChild;
            }
            else
            {
                /* we've hit the bottom */
                tree[here].rightChild = charIndex;
                tree[charIndex].leftChild = NULL_INDEX;
                tree[charIndex].rightChild = NULL_INDEX;
                tree[charIndex].parent = here;
                FixChildren(charIndex);
                return;
            }
        }
        else
        {
            /* replace old with new */
            tree[charIndex].leftChild = tree[here].leftChild;
            tree[charIndex].rightChild = tree[here].rightChild;
            tree[charIndex].parent = tree[here].parent;
            FixChildren(charIndex);

            if (tree[tree[here].parent].leftChild == here)
            {
                tree[tree[here].parent].leftChild = charIndex;
            }
            else
            {
                tree[tree[here].parent].rightChild = charIndex;
            }

            /* remove old node from the tree */
            tree[here].leftChild = NULL_INDEX;
            tree[here].rightChild = NULL_INDEX;
            tree[here].parent = NULL_INDEX;
            return;
        }

        compare = CompareString(charIndex, here);
    }
}

/****************************************************************************
*   Function   : AddString
*   Description: This function removes the MAX_UNCODED long string starting
*                at slidingWindow[charIndex] from the binary tree.
*   Parameters : charIndex - sliding window index of the string to be
*                            removed from the binary tree list.
*   Effects    : The string starting at slidingWindow[charIndex] is removed
*                from the sorted binary tree.
*   Returned   : NONE
****************************************************************************/
void RemoveString(unsigned int charIndex)
{
    unsigned int here;

    if (NULL_INDEX == tree[charIndex].parent)
    {
        return;     /* string isn't in tree */
    }

    if (NULL_INDEX == tree[charIndex].rightChild)
    {
        /* node doesn't have right child.  promote left child. */
        here = tree[charIndex].leftChild;
    }
    else if (NULL_INDEX == tree[charIndex].leftChild)
    {
        /* node doesn't have left child.  promote right child. */
        here = tree[charIndex].rightChild;
    }
    else
    {
        /* promote rightmost left descendant */
        here = tree[charIndex].leftChild;

        while (tree[here].rightChild != NULL_INDEX)
        {
            here = tree[here].rightChild;
        }

        if (here != tree[charIndex].leftChild)
        {
            /* there was a right branch to follow and we're at the end */
            tree[tree[here].parent].rightChild = tree[here].leftChild;
            tree[tree[here].leftChild].parent = tree[here].parent;
            tree[here].leftChild = tree[charIndex].leftChild;
            tree[tree[charIndex].leftChild].parent = here;
        }

        tree[here].rightChild = tree[charIndex].rightChild;
        tree[tree[charIndex].rightChild].parent = here;
    }

    if (tree[tree[charIndex].parent].leftChild == charIndex)
    {
        tree[tree[charIndex].parent].leftChild = here;
    }
    else
    {
        tree[tree[charIndex].parent].rightChild = here;
    }

    tree[here].parent = tree[charIndex].parent;

    if (treeRoot == charIndex)
    {
        treeRoot = here;
    }

    /* clear all pointers in deleted node. */
    tree[charIndex].leftChild = NULL_INDEX;
    tree[charIndex].rightChild = NULL_INDEX;
    tree[charIndex].parent = NULL_INDEX;
}

/****************************************************************************
*   Function   : ReplaceChar
*   Description: This function replaces the character stored in
*                slidingWindow[charIndex] with the one specified by
*                replacement.  The binary tree entries effected by the
*                replacement are also corrected.
*   Parameters : charIndex - sliding window index of the character to be
*                            removed from the linked list.
*   Effects    : slidingWindow[charIndex] is replaced by replacement.  Old
*                binary tree nodes for strings containing
*                slidingWindow[charIndex] are removed and new ones are
*                added.
*   Returned   : NONE
****************************************************************************/
void ReplaceChar(unsigned int charIndex, unsigned char replacement)
{
    unsigned int firstIndex, i;

    if (charIndex < MAX_UNCODED)
    {
        firstIndex = (WINDOW_SIZE + charIndex) - MAX_UNCODED;
    }
    else
    {
        firstIndex = charIndex - MAX_UNCODED;
    }

    /* remove all hash entries containing character at char index */
    for (i = 0; i < (MAX_UNCODED + 1); i++)
    {
        RemoveString(Wrap((firstIndex + i), WINDOW_SIZE));
    }

    slidingWindow[charIndex] = replacement;

    /* add all hash entries containing character at char index */
    for (i = 0; i < (MAX_UNCODED + 1); i++)
    {
        AddString(Wrap((firstIndex + i), WINDOW_SIZE));
    }
}

/****************************************************************************
*   Function   : PrintLen
*   Description: This function prints the string of length len that starts at
*                slidingWindow[charIndex].
*   Parameters : charIndex - sliding window index of the string to be
*                            printed.
*                len - length of the string to be printed.
*   Effects    : The string of length len starting at
*                slidingWindow[charIndex] is printed to stdout.
*   Returned   : NONE
****************************************************************************/
void PrintLen(unsigned int charIndex, unsigned int len)
{
    unsigned int i;

    for (i = 0; i < len; i++)
    {
        if (isprint(slidingWindow[Wrap((i + charIndex), WINDOW_SIZE)]))
        {
            putchar(slidingWindow[Wrap((i + charIndex), WINDOW_SIZE)]);
        }
        else
        {
            printf("<%02X>", slidingWindow[Wrap((i + charIndex), WINDOW_SIZE)]);
        }
    }
}

/****************************************************************************
*   Function   : DumpTree
*   Description: This function dumps the contents of the (sub)tree starting
*                at node 'root' to stdout.
*   Parameters : root - root node for subtree to be dumped
*   Effects    : The nodes contents of the (sub)tree rooted at node 'root' 
*                are printed to stdout.
*   Returned   : NONE
****************************************************************************/
void DumpTree(unsigned int root)
{
    if (NULL_INDEX == root)
    {
        /* empty tree */
        return;
    }

    if (tree[root].leftChild != NULL_INDEX)
    {
        DumpTree(tree[root].leftChild);
    }

    printf("%03d: ", root);
    PrintLen(root, MAX_CODED);
    printf("\n");

    if (tree[root].rightChild != NULL_INDEX)
    {
        DumpTree(tree[root].rightChild);
    }
}

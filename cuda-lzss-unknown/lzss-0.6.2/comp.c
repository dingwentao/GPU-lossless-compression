/***************************************************************************
*                     Sample Program Using LZSS Library
*
*   File    : comp.c
*   Purpose : Demonstrate usage of LZSS library to compress a file
*   Author  : Michael Dipperstein
*   Date    : November 7, 2004
*
****************************************************************************
*   UPDATES
*
*   $Id: comp.c,v 1.3 2007/09/20 04:34:25 michael Exp $
*   $Log: comp.c,v $
*   Revision 1.3  2007/09/20 04:34:25  michael
*   Changes required for LGPL v3.
*
*   Revision 1.2  2006/12/26 04:09:09  michael
*   Updated e-mail address and minor text clean-up.
*
*   Revision 1.1  2004/11/08 06:06:07  michael
*   Initial revision of super simple compression and decompression samples.
*
*
****************************************************************************
*
* COMP: Super simple sample demonstraiting use of LZSS compression routine.
* Copyright (C) 2004, 2006, 2007 by
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
#include "lzss.h"

/***************************************************************************
*                                FUNCTIONS
***************************************************************************/

/****************************************************************************
*   Function   : main
*   Description: This is the main function for this program it calls the
*                LZSS encoding routine using the command line arguments as
*                the file to encode and the encoded target.
*   Parameters : argc - number of parameters
*                argv - parameter list
*   Effects    : Encodes specified file
*   Returned   : status from EncodeLZSS
****************************************************************************/
int main(int argc, char *argv[])
{
    if (argc != 3)
    {
        printf("Invalid arguments\n");
        printf("Correct format:\n");
        printf("%s <compressed file> <decompressed file>\n", argv[0]);

        return 0;
    }

    return EncodeLZSS(argv[1], argv[2]);
}

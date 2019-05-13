############################################################################
# Makefile for lzss encode/decode library and sample program
#
#   $Id: Makefile,v 1.6 2010/02/12 05:18:41 michael Exp $
#   $Log: Makefile,v $
#   Revision 1.6  2010/02/12 05:18:41  michael
#   Include implementation of the Knuth-Morris-Pratt search optimization.
#
#   Revision 1.5  2007/09/20 04:34:45  michael
#   Replace getopt with optlist.
#   Changes required for LGPL v3.
#
#   Revision 1.4  2007/07/16 02:16:15  michael
#   Use -pedantic option when compiling.
#
#   Revision 1.3  2004/11/08 05:54:18  michael
#   1. Split encode and decode routines for smarter linking
#   2. Renamed lzsample.c sample.c to match my other samples
#   3. Makefile now builds code as libraries for better LGPL compliance.
#
#   Revision 1.2  2004/02/22 17:12:41  michael
#   - Separated encode/decode, match finding, and main.
#   - Use bitfiles for reading/writing files
#   - Use traditional LZSS encoding where the coded/uncoded bits
#     precede the symbol they are associated with, rather than
#     aggregating the bits.
#
#   Revision 1.1.1.1  2004/01/21 06:25:49  michael
#   Initial version
#
#
############################################################################
CC = gcc
LD = gcc
CFLAGS = -I. -O3 -Wall -pedantic -ansi -c
LDFLAGS = -O3 -o

# libraries
LIBS = -L. -llzss -loptlist

# Treat NT and non-NT windows the same
ifeq ($(OS),Windows_NT)
	OS = Windows
endif

ifeq ($(OS),Windows)
	ifeq ($(OSTYPE), cygwin)
		EXE = .exe
		DEL = rm
	else
		EXE = .exe
		DEL = del
	endif
else	#assume Linux/Unix
	EXE =
	DEL = rm
endif

# define the method to be used for searching for matches (choose one)
# brute force
FMOBJ = brute.o
# linked list
# FMOBJ = list.o
# hash table
# FMOBJ = hash.o
# Knuth–Morris–Pratt search
# FMOBJ = kmp.o
# binary tree
# FMOBJ = tree.o

LZOBJS = $(FMOBJ) lzencode.o lzvars.o

all:		sample$(EXE) liblzss.a liboptlist.a

sample$(EXE):	sample.o liblzss.a liboptlist.a
		$(LD) $< $(LIBS) $(LDFLAGS) $@

sample.o:	sample.c lzss.h optlist.h
		$(CC) $(CFLAGS) $<

liblzss.a:	$(LZOBJS) bitfile.o
		ar crv liblzss.a $(LZOBJS) bitfile.o
		ranlib liblzss.a

lzencode.o:	lzencode.c lzlocal.h bitfile.h
		$(CC) $(CFLAGS) $<

brute.o:	brute.c lzlocal.h
		$(CC) $(CFLAGS) $<

list.o:		list.c lzlocal.h
		$(CC) $(CFLAGS) $<

lzvars.o:	lzvars.c lzlocal.h
		$(CC) $(CFLAGS) $<

bitfile.o:	bitfile.c bitfile.h
		$(CC) $(CFLAGS) $<

liboptlist.a:	optlist.o
		ar crv liboptlist.a optlist.o
		ranlib liboptlist.a

optlist.o:	optlist.c optlist.h
		$(CC) $(CFLAGS) $<
clean:
		$(DEL) *.o
		$(DEL) *.a
		$(DEL) sample$(EXE)


/*-------------------------------------------------------------*/
/*--- Compression machinery (not incl block sorting)        ---*/
/*---                                            compress.c ---*/
/*-------------------------------------------------------------*/

/* ------------------------------------------------------------------
   This file is part of bzip2/libbzip2, a program and library for
   lossless, block-sorting data compression.

   bzip2/libbzip2 version 1.0.6 of 6 September 2010
   Copyright (C) 1996-2010 Julian Seward <jseward@bzip.org>

   Please read the WARNING, DISCLAIMER and PATENTS sections in the 
   README file.

   This program is released under the terms of the license contained
   in the file LICENSE.
   ------------------------------------------------------------------ */


/* CHANGES
    0.9.0    -- original version.
    0.9.0a/b -- no changes in this file.
    0.9.0c   -- changed setting of nGroups in sendMTFValues() 
                so as to do a bit better on small files
*/

#include "bzlib_private.h"
#include<omp.h>
#include<pthread.h>
#include<semaphore.h>


/*---------------------------------------------------*/
/*--- Bit stream I/O                              ---*/
/*---------------------------------------------------*/

/*---------------------------------------------------*/
void BZ2_bsInitWrite ( EState* s )
{
   s->bsLive = 0;
   s->bsBuff = 0;
}


/*---------------------------------------------------*/
static
void bsFinishWrite ( EState* s )
{
   while (s->bsLive > 0) {
      s->zbits[s->numZ] = (UChar)(s->bsBuff >> 24);
      s->numZ++;
      s->bsBuff <<= 8;
      s->bsLive -= 8;
   }
}


/*---------------------------------------------------*/
#define bsNEEDW(nz)                           \
{                                             \
   while (s->bsLive >= 8) {                   \
      s->zbits[s->numZ]                       \
         = (UChar)(s->bsBuff >> 24);          \
      s->numZ++;                              \
      s->bsBuff <<= 8;                        \
      s->bsLive -= 8;                         \
   }                                          \
}


/*---------------------------------------------------*/
static
__inline__
void bsW ( EState* s, Int32 n, UInt32 v )
{
   bsNEEDW ( n );
   s->bsBuff |= (v << (32 - s->bsLive - n));
   s->bsLive += n;
}


/*---------------------------------------------------*/
static
void bsPutUInt32 ( EState* s, UInt32 u )
{
   bsW ( s, 8, (u >> 24) & 0xffL );
   bsW ( s, 8, (u >> 16) & 0xffL );
   bsW ( s, 8, (u >>  8) & 0xffL );
   bsW ( s, 8,  u        & 0xffL );
}


/*---------------------------------------------------*/
static
void bsPutUChar ( EState* s, UChar c )
{
   bsW( s, 8, (UInt32)c );
}


/*---------------------------------------------------*/
/*--- The back end proper                         ---*/
/*---------------------------------------------------*/

/*---------------------------------------------------*/
static
void makeMaps_e ( EState* s )
{
   Int32 i;
   s->nInUse = 0;
   for (i = 0; i < 256; i++)
      if (s->inUse[i]) {
         s->unseqToSeq[i] = s->nInUse;
         s->nInUse++;
      }
}


/*---------------------------------------------------*/
static
void generateMTFValues ( EState* s )
{
   UChar   yy[256];
   Int32   i, j;
   Int32   zPend;
   Int32   wr;
   Int32   EOB;

   /* 
      After sorting (eg, here),
         s->arr1 [ 0 .. s->nblock-1 ] holds sorted order,
         and
         ((UChar*)s->arr2) [ 0 .. s->nblock-1 ] 
         holds the original block data.

      The first thing to do is generate the MTF values,
      and put them in
         ((UInt16*)s->arr1) [ 0 .. s->nblock-1 ].
      Because there are strictly fewer or equal MTF values
      than block values, ptr values in this area are overwritten
      with MTF values only when they are no longer needed.

      The final compressed bitstream is generated into the
      area starting at
         (UChar*) (&((UChar*)s->arr2)[s->nblock])

      These storage aliases are set up in bzCompressInit(),
      except for the last one, which is arranged in 
      compressBlock().
   */
   UInt32* ptr   = s->ptr;
   UChar* block  = s->block;
   UInt16* mtfv  = s->mtfv;

   makeMaps_e ( s );
   EOB = s->nInUse+1;

   for (i = 0; i <= EOB; i++) s->mtfFreq[i] = 0;

   wr = 0;
   zPend = 0;
   for (i = 0; i < s->nInUse; i++) yy[i] = (UChar) i;

   for (i = 0; i < s->nblock; i++) {
      UChar ll_i;
      AssertD ( wr <= i, "generateMTFValues(1)" );
      j = ptr[i]-1; if (j < 0) j += s->nblock;
      ll_i = s->unseqToSeq[block[j]];
      AssertD ( ll_i < s->nInUse, "generateMTFValues(2a)" );

      if (yy[0] == ll_i) { 
         zPend++;
      } else {

         if (zPend > 0) {
            zPend--;
            while (True) {
               if (zPend & 1) {
                  mtfv[wr] = BZ_RUNB; wr++; 
                  s->mtfFreq[BZ_RUNB]++; 
               } else {
                  mtfv[wr] = BZ_RUNA; wr++; 
                  s->mtfFreq[BZ_RUNA]++; 
               }
               if (zPend < 2) break;
               zPend = (zPend - 2) / 2;
            };
            zPend = 0;
         }
         {
            register UChar  rtmp;
            register UChar* ryy_j;
            register UChar  rll_i;
            rtmp  = yy[1];
            yy[1] = yy[0];
            ryy_j = &(yy[1]);
            rll_i = ll_i;
            while ( rll_i != rtmp ) {
               register UChar rtmp2;
               ryy_j++;
               rtmp2  = rtmp;
               rtmp   = *ryy_j;
               *ryy_j = rtmp2;
            };
            yy[0] = rtmp;
            j = ryy_j - &(yy[0]);
            mtfv[wr] = j+1; wr++; s->mtfFreq[j+1]++;
         }

      }
   }

   if (zPend > 0) {
      zPend--;
      while (True) {
         if (zPend & 1) {
            mtfv[wr] = BZ_RUNB; wr++; 
            s->mtfFreq[BZ_RUNB]++; 
         } else {
            mtfv[wr] = BZ_RUNA; wr++; 
            s->mtfFreq[BZ_RUNA]++; 
         }
         if (zPend < 2) break;
         zPend = (zPend - 2) / 2;
      };
      zPend = 0;
   }

   mtfv[wr] = EOB; wr++; s->mtfFreq[EOB]++;

   s->nMTF = wr;
}


/*---------------------------------------------------*/
#define BZ_LESSER_ICOST  0
#define BZ_GREATER_ICOST 15

static
void sendMTFValues ( EState* s )
{
   Int32 v, t, i, j, gs, ge, totc, bt, bc, iter;
   Int32 nSelectors, alphaSize, minLen, maxLen, selCtr;
   Int32 nGroups, nBytes;

   /*--
   UChar  len [BZ_N_GROUPS][BZ_MAX_ALPHA_SIZE];
   is a global since the decoder also needs it.

   Int32  code[BZ_N_GROUPS][BZ_MAX_ALPHA_SIZE];
   Int32  rfreq[BZ_N_GROUPS][BZ_MAX_ALPHA_SIZE];
   are also globals only used in this proc.
   Made global to keep stack frame size small.
   --*/


   UInt16 cost[BZ_N_GROUPS];
   Int32  fave[BZ_N_GROUPS];

   UInt16* mtfv = s->mtfv;

   if (s->verbosity >= 3)
      VPrintf3( "      %d in block, %d after MTF & 1-2 coding, "
                "%d+2 syms in use\n", 
                s->nblock, s->nMTF, s->nInUse );

   alphaSize = s->nInUse+2;
   for (t = 0; t < BZ_N_GROUPS; t++)
      for (v = 0; v < alphaSize; v++)
         s->len[t][v] = BZ_GREATER_ICOST;

   /*--- Decide how many coding tables to use ---*/
   AssertH ( s->nMTF > 0, 3001 );
   if (s->nMTF < 200)  nGroups = 2; else
   if (s->nMTF < 600)  nGroups = 3; else
   if (s->nMTF < 1200) nGroups = 4; else
   if (s->nMTF < 2400) nGroups = 5; else
                       nGroups = 6;

   /*--- Generate an initial set of coding tables ---*/
   { 
      Int32 nPart, remF, tFreq, aFreq;

      nPart = nGroups;
      remF  = s->nMTF;
      gs = 0;
      while (nPart > 0) {
         tFreq = remF / nPart;
         ge = gs-1;
         aFreq = 0;
         while (aFreq < tFreq && ge < alphaSize-1) {
            ge++;
            aFreq += s->mtfFreq[ge];
         }

         if (ge > gs 
             && nPart != nGroups && nPart != 1 
             && ((nGroups-nPart) % 2 == 1)) {
            aFreq -= s->mtfFreq[ge];
            ge--;
         }

         if (s->verbosity >= 3)
            VPrintf5( "      initial group %d, [%d .. %d], "
                      "has %d syms (%4.1f%%)\n",
                      nPart, gs, ge, aFreq, 
                      (100.0 * (float)aFreq) / (float)(s->nMTF) );
 
         for (v = 0; v < alphaSize; v++)
            if (v >= gs && v <= ge) 
               s->len[nPart-1][v] = BZ_LESSER_ICOST; else
               s->len[nPart-1][v] = BZ_GREATER_ICOST;
 
         nPart--;
         gs = ge+1;
         remF -= aFreq;
      }
   }

   /*--- 
      Iterate up to BZ_N_ITERS times to improve the tables.
   ---*/
   for (iter = 0; iter < BZ_N_ITERS; iter++) {

      for (t = 0; t < nGroups; t++) fave[t] = 0;

      for (t = 0; t < nGroups; t++)
         for (v = 0; v < alphaSize; v++)
            s->rfreq[t][v] = 0;

      /*---
        Set up an auxiliary length table which is used to fast-track
	the common case (nGroups == 6). 
      ---*/
      if (nGroups == 6) {
         for (v = 0; v < alphaSize; v++) {
            s->len_pack[v][0] = (s->len[1][v] << 16) | s->len[0][v];
            s->len_pack[v][1] = (s->len[3][v] << 16) | s->len[2][v];
            s->len_pack[v][2] = (s->len[5][v] << 16) | s->len[4][v];
	 }
      }

      nSelectors = 0;
      totc = 0;
      gs = 0;
      while (True) {

         /*--- Set group start & end marks. --*/
         if (gs >= s->nMTF) break;
         ge = gs + BZ_G_SIZE - 1; 
         if (ge >= s->nMTF) ge = s->nMTF-1;

         /*-- 
            Calculate the cost of this group as coded
            by each of the coding tables.
         --*/
         for (t = 0; t < nGroups; t++) cost[t] = 0;

         if (nGroups == 6 && 50 == ge-gs+1) {
            /*--- fast track the common case ---*/
            register UInt32 cost01, cost23, cost45;
            register UInt16 icv;
            cost01 = cost23 = cost45 = 0;

#           define BZ_ITER(nn)                \
               icv = mtfv[gs+(nn)];           \
               cost01 += s->len_pack[icv][0]; \
               cost23 += s->len_pack[icv][1]; \
               cost45 += s->len_pack[icv][2]; \

            BZ_ITER(0);  BZ_ITER(1);  BZ_ITER(2);  BZ_ITER(3);  BZ_ITER(4);
            BZ_ITER(5);  BZ_ITER(6);  BZ_ITER(7);  BZ_ITER(8);  BZ_ITER(9);
            BZ_ITER(10); BZ_ITER(11); BZ_ITER(12); BZ_ITER(13); BZ_ITER(14);
            BZ_ITER(15); BZ_ITER(16); BZ_ITER(17); BZ_ITER(18); BZ_ITER(19);
            BZ_ITER(20); BZ_ITER(21); BZ_ITER(22); BZ_ITER(23); BZ_ITER(24);
            BZ_ITER(25); BZ_ITER(26); BZ_ITER(27); BZ_ITER(28); BZ_ITER(29);
            BZ_ITER(30); BZ_ITER(31); BZ_ITER(32); BZ_ITER(33); BZ_ITER(34);
            BZ_ITER(35); BZ_ITER(36); BZ_ITER(37); BZ_ITER(38); BZ_ITER(39);
            BZ_ITER(40); BZ_ITER(41); BZ_ITER(42); BZ_ITER(43); BZ_ITER(44);
            BZ_ITER(45); BZ_ITER(46); BZ_ITER(47); BZ_ITER(48); BZ_ITER(49);

#           undef BZ_ITER

            cost[0] = cost01 & 0xffff; cost[1] = cost01 >> 16;
            cost[2] = cost23 & 0xffff; cost[3] = cost23 >> 16;
            cost[4] = cost45 & 0xffff; cost[5] = cost45 >> 16;

         } else {
	    /*--- slow version which correctly handles all situations ---*/
            for (i = gs; i <= ge; i++) { 
               UInt16 icv = mtfv[i];
               for (t = 0; t < nGroups; t++) cost[t] += s->len[t][icv];
            }
         }
 
         /*-- 
            Find the coding table which is best for this group,
            and record its identity in the selector table.
         --*/
         bc = 999999999; bt = -1;
         for (t = 0; t < nGroups; t++)
            if (cost[t] < bc) { bc = cost[t]; bt = t; };
         totc += bc;
         fave[bt]++;
         s->selector[nSelectors] = bt;
         nSelectors++;

         /*-- 
            Increment the symbol frequencies for the selected table.
          --*/
         if (nGroups == 6 && 50 == ge-gs+1) {
            /*--- fast track the common case ---*/

#           define BZ_ITUR(nn) s->rfreq[bt][ mtfv[gs+(nn)] ]++

            BZ_ITUR(0);  BZ_ITUR(1);  BZ_ITUR(2);  BZ_ITUR(3);  BZ_ITUR(4);
            BZ_ITUR(5);  BZ_ITUR(6);  BZ_ITUR(7);  BZ_ITUR(8);  BZ_ITUR(9);
            BZ_ITUR(10); BZ_ITUR(11); BZ_ITUR(12); BZ_ITUR(13); BZ_ITUR(14);
            BZ_ITUR(15); BZ_ITUR(16); BZ_ITUR(17); BZ_ITUR(18); BZ_ITUR(19);
            BZ_ITUR(20); BZ_ITUR(21); BZ_ITUR(22); BZ_ITUR(23); BZ_ITUR(24);
            BZ_ITUR(25); BZ_ITUR(26); BZ_ITUR(27); BZ_ITUR(28); BZ_ITUR(29);
            BZ_ITUR(30); BZ_ITUR(31); BZ_ITUR(32); BZ_ITUR(33); BZ_ITUR(34);
            BZ_ITUR(35); BZ_ITUR(36); BZ_ITUR(37); BZ_ITUR(38); BZ_ITUR(39);
            BZ_ITUR(40); BZ_ITUR(41); BZ_ITUR(42); BZ_ITUR(43); BZ_ITUR(44);
            BZ_ITUR(45); BZ_ITUR(46); BZ_ITUR(47); BZ_ITUR(48); BZ_ITUR(49);

#           undef BZ_ITUR

         } else {
	    /*--- slow version which correctly handles all situations ---*/
            for (i = gs; i <= ge; i++)
               s->rfreq[bt][ mtfv[i] ]++;
         }

         gs = ge+1;
      }
      if (s->verbosity >= 3) {
         VPrintf2 ( "      pass %d: size is %d, grp uses are ", 
                   iter+1, totc/8 );
         for (t = 0; t < nGroups; t++)
            VPrintf1 ( "%d ", fave[t] );
         VPrintf0 ( "\n" );
      }

      /*--
        Recompute the tables based on the accumulated frequencies.
      --*/
      /* maxLen was changed from 20 to 17 in bzip2-1.0.3.  See 
         comment in huffman.c for details. */
      for (t = 0; t < nGroups; t++)
         BZ2_hbMakeCodeLengths ( &(s->len[t][0]), &(s->rfreq[t][0]), 
                                 alphaSize, 17 /*20*/ );
   }


   AssertH( nGroups < 8, 3002 );


   AssertH( /* nSelectors < 32768 &&  */
            nSelectors <= BZ_MAX_SELECTORS,
            3003 );


   /*--- Compute MTF values for the selectors. ---*/
   {
      UChar pos[BZ_N_GROUPS], ll_i, tmp2, tmp;
      for (i = 0; i < nGroups; i++) pos[i] = i;
      for (i = 0; i < nSelectors; i++) {
         ll_i = s->selector[i];
         j = 0;
         tmp = pos[j];
         while ( ll_i != tmp ) {
            j++;
            tmp2 = tmp;
            tmp = pos[j];
            pos[j] = tmp2;
         };
         pos[0] = tmp;
         s->selectorMtf[i] = j;
      }
   };

   /*--- Assign actual codes for the tables. --*/
   for (t = 0; t < nGroups; t++) {
      minLen = 32;
      maxLen = 0;
      for (i = 0; i < alphaSize; i++) {
         if (s->len[t][i] > maxLen) maxLen = s->len[t][i];
         if (s->len[t][i] < minLen) minLen = s->len[t][i];
      }
      AssertH ( !(maxLen > 17 /*20*/ ), 3004 );
      AssertH ( !(minLen < 1),  3005 );
      BZ2_hbAssignCodes ( &(s->code[t][0]), &(s->len[t][0]), 
                          minLen, maxLen, alphaSize );
   }

   /*--- Transmit the mapping table. ---*/
   { 
      Bool inUse16[16];
      for (i = 0; i < 16; i++) {
          inUse16[i] = False;
          for (j = 0; j < 16; j++)
             if (s->inUse[i * 16 + j]) inUse16[i] = True;
      }
     
      nBytes = s->numZ;
      for (i = 0; i < 16; i++)
         if (inUse16[i]) bsW(s,1,1); else bsW(s,1,0);

      for (i = 0; i < 16; i++)
         if (inUse16[i])
            for (j = 0; j < 16; j++) {
               if (s->inUse[i * 16 + j]) bsW(s,1,1); else bsW(s,1,0);
            }

      if (s->verbosity >= 3) 
         VPrintf1( "      bytes: mapping %d, ", s->numZ-nBytes );
   }

   /*--- Now the selectors. ---*/
   nBytes = s->numZ;
   bsW ( s, 5 /* 3 */, nGroups ); // changed @aditya


   bsW ( s, 17 /* 15 */, nSelectors ); // changed @aditya
   
   for (i = 0; i < nSelectors; i++) { 
      for (j = 0; j < s->selectorMtf[i]; j++) bsW(s,1,1);
      bsW(s,1,0);
   }
   if (s->verbosity >= 3)
      VPrintf1( "selectors %d, ", s->numZ-nBytes );

   /*--- Now the coding tables. ---*/
   nBytes = s->numZ;

   for (t = 0; t < nGroups; t++) {
      Int32 curr = s->len[t][0];
      bsW ( s, 5, curr );
      for (i = 0; i < alphaSize; i++) {
         while (curr < s->len[t][i]) { bsW(s,2,2); curr++; /* 10 */ };
         while (curr > s->len[t][i]) { bsW(s,2,3); curr--; /* 11 */ };
         bsW ( s, 1, 0 );
      }
   }

   if (s->verbosity >= 3)
      VPrintf1 ( "code lengths %d, ", s->numZ-nBytes );

   /*--- And finally, the block data proper ---*/
   nBytes = s->numZ;
   selCtr = 0;
   gs = 0;
   while (True) {
      if (gs >= s->nMTF) break;
      ge = gs + BZ_G_SIZE - 1; 
      if (ge >= s->nMTF) ge = s->nMTF-1;
      AssertH ( s->selector[selCtr] < nGroups, 3006 );

      if (nGroups == 6 && 50 == ge-gs+1) {
            /*--- fast track the common case ---*/
            UInt16 mtfv_i;
            UChar* s_len_sel_selCtr 
               = &(s->len[s->selector[selCtr]][0]);
            Int32* s_code_sel_selCtr
               = &(s->code[s->selector[selCtr]][0]);

#           define BZ_ITAH(nn)                      \
               mtfv_i = mtfv[gs+(nn)];              \
               bsW ( s,                             \
                     s_len_sel_selCtr[mtfv_i],      \
                     s_code_sel_selCtr[mtfv_i] )

            BZ_ITAH(0);  BZ_ITAH(1);  BZ_ITAH(2);  BZ_ITAH(3);  BZ_ITAH(4);
            BZ_ITAH(5);  BZ_ITAH(6);  BZ_ITAH(7);  BZ_ITAH(8);  BZ_ITAH(9);
            BZ_ITAH(10); BZ_ITAH(11); BZ_ITAH(12); BZ_ITAH(13); BZ_ITAH(14);
            BZ_ITAH(15); BZ_ITAH(16); BZ_ITAH(17); BZ_ITAH(18); BZ_ITAH(19);
            BZ_ITAH(20); BZ_ITAH(21); BZ_ITAH(22); BZ_ITAH(23); BZ_ITAH(24);
            BZ_ITAH(25); BZ_ITAH(26); BZ_ITAH(27); BZ_ITAH(28); BZ_ITAH(29);
            BZ_ITAH(30); BZ_ITAH(31); BZ_ITAH(32); BZ_ITAH(33); BZ_ITAH(34);
            BZ_ITAH(35); BZ_ITAH(36); BZ_ITAH(37); BZ_ITAH(38); BZ_ITAH(39);
            BZ_ITAH(40); BZ_ITAH(41); BZ_ITAH(42); BZ_ITAH(43); BZ_ITAH(44);
            BZ_ITAH(45); BZ_ITAH(46); BZ_ITAH(47); BZ_ITAH(48); BZ_ITAH(49);

#           undef BZ_ITAH

      } else {
	 /*--- slow version which correctly handles all situations ---*/
         for (i = gs; i <= ge; i++) {
            bsW ( s, 
                  s->len  [s->selector[selCtr]] [mtfv[i]],
                  s->code [s->selector[selCtr]] [mtfv[i]] );
         }
      }


      gs = ge+1;
      selCtr++;
   }
   AssertH( selCtr == nSelectors, 3007 );

   if (s->verbosity >= 3)
      VPrintf1( "codes %d\n", s->numZ-nBytes );
}


void merge_two_sort_arrays ( EState *s ) 
{ 

	unsigned char *block = (unsigned char*) s->arr2;
	unsigned int *h_first_sort_rank = (unsigned int*) s->arr1_first_sort_rank;
	unsigned int *h_first_sort_index = (unsigned int*) s->arr1_first_sort;
	unsigned int *h_second_sort_index = (unsigned int*) s->arr1_second_sort;
	unsigned int *order = (unsigned int*) s->arr1;

	/* stores position of index 0 in BWT transform */
	s->origPtr = -1;
	
	int originalLength = s->nblock;
	int firstSortLength = s->first_sort_length;
	int secondSortLength = originalLength - firstSortLength;

	int countOrderArr = 0;
	int countFirstArr = 0;
	int countSecondArr = 0;
	int indexFirst;
	int indexSecond;

	for(countOrderArr = 0; countOrderArr < originalLength && countFirstArr < firstSortLength && countSecondArr < secondSortLength; countOrderArr++) { 
		indexFirst = h_first_sort_index[countFirstArr];
		indexSecond = h_second_sort_index[countSecondArr];
		if(block[indexFirst] != block[indexSecond]) {
			block[indexFirst] < block[indexSecond] ? order[countOrderArr] = h_first_sort_index[countFirstArr++] : order[countOrderArr] = h_second_sort_index[countSecondArr++];
			continue;
		}

		if(indexFirst == originalLength - 1) { 
			if(block[0] == block[indexSecond + 1]) { 
				if(indexSecond == originalLength - 2) { 
					if(block[1]  == block[0]) { 
						h_first_sort_rank[2] < h_first_sort_rank[1] ? order[countOrderArr] = h_first_sort_index[countFirstArr++] : order[countOrderArr] = h_second_sort_index[countSecondArr++];
					} else { 
						block[1] < block[0] ? order[countOrderArr] = h_first_sort_index[countFirstArr++] : order[countOrderArr] = h_second_sort_index[countSecondArr++];
					}
				} else { 
					h_first_sort_rank[1] < h_first_sort_rank[indexSecond+2] ? order[countOrderArr] = h_first_sort_index[countFirstArr++] : order[countOrderArr] = h_second_sort_index[countSecondArr++];
				}
			} else { 
				block[0] < block[indexSecond+1] ? order[countOrderArr] = h_first_sort_index[countFirstArr++] : order[countOrderArr] = h_second_sort_index[countSecondArr++];
			}
		} else if(indexFirst % 3 == 1) {
			(h_first_sort_rank[indexFirst + 1] < h_first_sort_rank[indexSecond + 1]) ? order[countOrderArr] = h_first_sort_index[countFirstArr++] : order[countOrderArr] = h_second_sort_index[countSecondArr++];
		
		} else if(indexFirst % 3 == 2) {
			if(block[indexFirst+1] == block[indexSecond+1]) {
				if(indexFirst + 2 == originalLength || indexSecond+2 == originalLength) { 
					int itrIndexFirst = (indexFirst+2) % originalLength;
					int itrIndexSecond = (indexSecond+2) % originalLength;
					int foundDifference = 0;
					while(itrIndexFirst % 3 == 0 || itrIndexSecond % 3 == 0) { 
						if(block[itrIndexFirst] != block[itrIndexSecond]) { 
							block[itrIndexFirst] < block[itrIndexSecond] ? order[countOrderArr] = h_first_sort_index[countFirstArr++] : order[countOrderArr] = h_second_sort_index[countSecondArr++];
							foundDifference = 1;
							break;
						}
						itrIndexFirst = (itrIndexFirst+1)%originalLength;
						itrIndexSecond = (itrIndexSecond+1)%originalLength;
					}
					if(foundDifference == 1) { 
						continue;
					}
					h_first_sort_rank[itrIndexFirst] < h_first_sort_rank[itrIndexSecond] ? order[countOrderArr] = h_first_sort_index[countFirstArr++] : order[countOrderArr] = h_second_sort_index[countSecondArr++];
				} else { 
					(h_first_sort_rank[indexFirst+2] < h_first_sort_rank[indexSecond+2]) ? order[countOrderArr] = h_first_sort_index[countFirstArr++] : order[countOrderArr] = h_second_sort_index[countSecondArr++];
				}
			} else { 
				(block[indexFirst+1] < block[indexSecond+1]) ? order[countOrderArr] = h_first_sort_index[countFirstArr++] : order[countOrderArr] = h_second_sort_index[countSecondArr++];
			}
		}

		if(order[countOrderArr] == 0) { 
			s->origPtr = countOrderArr;
		}
		
	} 
	
	while(countFirstArr < firstSortLength) { 
		order[countOrderArr] = h_first_sort_index[countFirstArr];
		if(order[countOrderArr] == 0) { 
			s->origPtr = countOrderArr;
		}
		countFirstArr++;
		countOrderArr++;
	}

	while(countSecondArr < secondSortLength) { 
		order[countOrderArr] = h_second_sort_index[countSecondArr];
		if(order[countOrderArr] == 0) { 
			s->origPtr = countOrderArr;
		}
		countSecondArr++;
		countOrderArr++;
	}  

	AssertH( s->origPtr != -1, 1003 ); 

	return;
}

Bool blocksort_wrapper( EState *s ) {
	BZ_INITIALISE_CRC( s->blockCRC );
	if (s->nblock > 0) {

		BZ_FINALISE_CRC ( s->blockCRC );
		s->combinedCRC = (s->combinedCRC << 1) | (s->combinedCRC >> 31);
		s->combinedCRC ^= s->blockCRC;
		if (s->blockNo > 1) s->numZ = 0;

		if (s->verbosity >= 2)
			VPrintf4( "    block %d: crc = 0x%08x, "
					"combined CRC = 0x%08x, size = %d\n",
					s->blockNo, s->blockCRC, s->combinedCRC, s->nblock );

		s->first_sort_length = gpuBlockSort( (UChar*) s->arr2, s->arr1, s->arr1_first_sort, s->arr1_second_sort, s->arr1_first_sort_rank, s->nblock, &(s->sortingDepth));
		
		/* loop moved to merge two arrays
		UInt32* ptr    = s->ptr; 
		s->origPtr = -1;
		Int32 i;
		for (i = 0; i < s->nblock; i++)
			if (ptr[i] == 0)
			{ s->origPtr = i; break; };

		AssertH( s->origPtr != -1, 1003 ); */
	}
	return True;
}

Bool mtf_huff_wrapper( EState* s, Bool is_last_block ) { 

	
	s->zbits = (UChar*) (&((UChar*)s->arr2)[s->nblock]);

#ifdef PRINT_DEBUG
	printf("BZ2_compressBlock %d\n",s->blockNo); 
#endif

	// start with a fresh buffer with every block
	BZ2_bsInitWrite( s ); 

	/*-- If this is the first block, create the stream header. --*/
	if (s->blockNo == 1) {
#ifdef PRINT_DEBUG
		printf("This is the first block\n");
#endif
		bsPutUChar ( s, BZ_HDR_B );
		bsPutUChar ( s, BZ_HDR_Z );
		bsPutUChar ( s, BZ_HDR_h );
		bsPutUChar ( s, (UChar)(BZ_HDR_0 + s->blockSize100k) );
	}

	if (s->nblock > 0) {

		bsPutUChar ( s, 0x31 ); bsPutUChar ( s, 0x41 );
		bsPutUChar ( s, 0x59 ); bsPutUChar ( s, 0x26 );
		bsPutUChar ( s, 0x53 ); bsPutUChar ( s, 0x59 );

		/*-- Now the block's CRC, so it is in a known place. --*/
		bsPutUInt32 ( s, s->blockCRC );

		/*-- 
		  Now a single bit indicating (non-)randomisation. 
		  As of version 0.9.5, we use a better sorting algorithm
		  which makes randomisation unnecessary.  So always set
		  the randomised bit to 'no'.  Of course, the decoder
		  still needs to be able to handle randomised blocks
		  so as to maintain backwards compatibility with
		  older versions of bzip2.
		  --*/
		bsW(s,1,0);

		bsW ( s, 24, s->origPtr );
		generateMTFValues ( s );
		sendMTFValues ( s );
	}


	/*-- If this is the last block, add the stream trailer. --*/
	if (is_last_block) {
#ifdef PRINT_DEBUG
		printf("This is the last block\n");
#endif
		bsPutUChar ( s, 0x17 ); bsPutUChar ( s, 0x72 );
		bsPutUChar ( s, 0x45 ); bsPutUChar ( s, 0x38 );
		bsPutUChar ( s, 0x50 ); bsPutUChar ( s, 0x90 );
		bsPutUInt32 ( s, s->combinedCRC );
		if (s->verbosity >= 2)
			VPrintf1( "    final combined CRC = 0x%08x\n   ", s->combinedCRC );
		bsFinishWrite ( s );
	}

	return True;
}

Bool BZ2_compressBlock_only_CPU ( EState* s, Bool is_last_block ) {

	if (s->nblock > 0) {
		BZ_FINALISE_CRC ( s->blockCRC );
		s->combinedCRC = (s->combinedCRC << 1) | (s->combinedCRC >> 31);
		s->combinedCRC ^= s->blockCRC;
		if (s->blockNo > 1) s->numZ = 0;

		if (s->verbosity >= 2)
			VPrintf4( "    block %d: crc = 0x%08x, "
					"combined CRC = 0x%08x, size = %d\n",
					s->blockNo, s->blockCRC, s->combinedCRC, s->nblock );

		BZ2_blockSort ( s );
	}

	s->zbits = (UChar*) (&((UChar*)s->arr2)[s->nblock]);

	/*-- If this is the first block, create the stream header. --*/
	BZ2_bsInitWrite ( s );
	if (s->blockNo == 1) {
		bsPutUChar ( s, BZ_HDR_B );
		bsPutUChar ( s, BZ_HDR_Z );
		bsPutUChar ( s, BZ_HDR_h );
		bsPutUChar ( s, (UChar)(BZ_HDR_0 + s->blockSize100k) );
	}

	if (s->nblock > 0) {

		bsPutUChar ( s, 0x31 ); bsPutUChar ( s, 0x41 );
		bsPutUChar ( s, 0x59 ); bsPutUChar ( s, 0x26 );
		bsPutUChar ( s, 0x53 ); bsPutUChar ( s, 0x59 );

		/*-- Now the block's CRC, so it is in a known place. --*/
		bsPutUInt32 ( s, s->blockCRC );

		/*-- 
		  Now a single bit indicating (non-)randomisation. 
		  As of version 0.9.5, we use a better sorting algorithm
		  which makes randomisation unnecessary.  So always set
		  the randomised bit to 'no'.  Of course, the decoder
		  still needs to be able to handle randomised blocks
		  so as to maintain backwards compatibility with
		  older versions of bzip2.
		  --*/
		bsW(s,1,0);

		bsW ( s, 24, s->origPtr );
		generateMTFValues ( s );
		sendMTFValues ( s );
	}


	/*-- If this is the last block, add the stream trailer. --*/
	if (is_last_block) {

		bsPutUChar ( s, 0x17 ); bsPutUChar ( s, 0x72 );
		bsPutUChar ( s, 0x45 ); bsPutUChar ( s, 0x38 );
		bsPutUChar ( s, 0x50 ); bsPutUChar ( s, 0x90 );
		bsPutUInt32 ( s, s->combinedCRC );
		if (s->verbosity >= 2)
			VPrintf1( "    final combined CRC = 0x%08x\n   ", s->combinedCRC );
		bsFinishWrite ( s );
	}

	return True;

}
/*---------------------------------------------------*/
void BZ2_compressBlocks ( bz_stream* strm)
{


	gpuSetDevice(0);
	
	struct timespec t1, t2; 
	clock_gettime(CLOCK_MONOTONIC, &t1);

	Int32 atomic_used_count = 1; 
	Int32 done_block_sort = 0;
	Int32 done_block_sort_queue[BZ_MAX_STATE_COUNT];
	Int32 done_huff_mtf = 0;
	Int32 finalGPUBlocks = BZ_MAX_STATE_COUNT;
	Int32 num_procs = omp_get_num_procs();

	printf("Block size : %d\n",((EState *)strm->state[0])->nblockMAX + 19);
	printf("numThreads read %d\n",((EState*)strm->state[0])->numThreads);
	Int32 num_threads = ((EState*)strm->state[0])->numThreads/*+2*/; 
	omp_set_num_threads(num_threads+2);
	printf("Number of additional CPU threads %d\n",num_threads);
	
        #pragma omp parallel shared(done_block_sort, done_block_sort_queue, done_huff_mtf, finalGPUBlocks)
	{ 
		int threadID = omp_get_thread_num();
		Int32 count;
		Int32 currentHuffMtf = -1;
		EState *s; 
		Bool is_last_block = False;

		if(threadID == 0) {
			struct timespec thread1_T1, thread1_T2; 
			clock_gettime(CLOCK_MONOTONIC, &thread1_T1);

			blocksort_wrapper((EState*) strm->state[0]);

			done_block_sort_queue[done_block_sort] = 0;
			done_block_sort++;
			while(true) {
				#pragma omp critical 
				{
					count = atomic_used_count;
					atomic_used_count++;
				}
				if(count > strm->state_fill_count) { 
					finalGPUBlocks = done_block_sort;
					break;
				}
				s = (EState *)strm->state[count];
				if(blocksort_wrapper( s )) {
					done_block_sort_queue[done_block_sort] = count;
					done_block_sort++; 
				} else { 
					printf("blocksort_wrapper failed for blockNo %d\n",count);
					exit(2);
				}
			}
			clock_gettime(CLOCK_MONOTONIC, &thread1_T2);
			double thread1_diff = thread1_T2.tv_sec - thread1_T1.tv_sec + ((thread1_T2.tv_nsec - thread1_T1.tv_nsec)/1000000000.0); 
			printf("[Time thread1] %lf\n",thread1_diff);
		}

		if(threadID == 1) {
			struct timespec thread2_T1, thread2_T2;
			clock_gettime(CLOCK_MONOTONIC, &thread2_T1);
			double thread2_work = 0.0;
			struct timespec thread2_work1, thread2_work2;
			while(true) {
				if(done_block_sort > done_huff_mtf) {
					clock_gettime(CLOCK_MONOTONIC, &thread2_work1);
				
					currentHuffMtf = done_block_sort_queue[done_huff_mtf];
					done_huff_mtf++; 
					s = (EState *)strm->state[currentHuffMtf];
					merge_two_sort_arrays ( s );
					if(currentHuffMtf == strm->state_fill_count) { 
						is_last_block = True;
					}
					if( !mtf_huff_wrapper( s, is_last_block ) ) { 
						printf("mtf_huff_wrapper failed for blockNo %d\n",currentHuffMtf);
						exit(2);
					}
				
					clock_gettime(CLOCK_MONOTONIC, &thread2_work2);
					thread2_work+= thread2_work2.tv_sec - thread2_work1.tv_sec + ((thread2_work2.tv_nsec - thread2_work1.tv_nsec)/1000000000.0);
				}
				if(done_huff_mtf == finalGPUBlocks) { 
					break;
				}
			} 
		
			clock_gettime(CLOCK_MONOTONIC, &thread2_T2);
			double thread2_diff = thread2_T2.tv_sec - thread2_T1.tv_sec + ((thread2_T2.tv_nsec - thread2_T1.tv_nsec)/1000000000.0); 
			printf("[Time thread2 work] %lf\n",thread2_work);
			printf("[Time thread2] %lf\n",thread2_diff);
		}

		if(threadID >= 2) {
			while(true) { 
				#pragma omp critical 
				{
					count = atomic_used_count;
					atomic_used_count++;
				}
				if(count > strm->state_fill_count) { 
					break;
				}
				s = (EState *)strm->state[count];
				if(!BZ2_compressBlock_only_CPU( s, (Bool) count == strm->state_fill_count)) { 
					printf("BZ2_compressBlock_only_CPU failed at blockNo %d\n",count);
					exit(2);
				}
			}
		} 
	} 

	/*
	UInt32 count;
	#pragma omp parallel for
	for(count = 0; count <= strm->state_fill_count; count++) { 
		EState *s = (EState*)strm->state[count];	
		if(!BZ2_compressBlock_only_CPU( s, (Bool) count == strm->state_fill_count)) { 
			printf("BZ2_compressBlock_only_CPU failed at blockNo %d\n",count);
			exit(2);
		}
	} */
	printf("Number of CPU threads %d\n",num_threads+2);
	printf("Out of the total %d blocks GPU did %d\n",strm->state_fill_count+1,finalGPUBlocks);
	clock_gettime(CLOCK_MONOTONIC, &t2);
	double t = t2.tv_sec - t1.tv_sec + ((t2.tv_nsec - t1.tv_nsec)/1000000000.0); 
	printf("total compression time (with overlap) %lf\n",t);
}


/* Global Variables for PThreads Producer-Consumer pipeline */
bz_stream* global_strm;
sem_t mutex, full, empty;
int buff[BZ_MAX_STATE_COUNT];
int producerCount = 0, consumerCount = 0;


void *produce(void *arg) { 
	int i;
	struct timespec t1, t2;
	double thread1_work = 0.0;

	int maxSortingDepth = 0;
	int avgSortingDepth = 0;

	for(i = 0; i <= global_strm->state_fill_count; i++) {
		 
		clock_gettime(CLOCK_MONOTONIC, &t1);
		EState *s = (EState*)global_strm->state[i];
		if( !blocksort_wrapper( s )) { 
			printf("[ERROR] blocksort_wrapper_failed\n");
			exit(1);
		}
		clock_gettime(CLOCK_MONOTONIC, &t2);
		if(i!=global_strm->state_fill_count) { 
			thread1_work += t2.tv_sec - t1.tv_sec + ((t2.tv_nsec - t1.tv_nsec)/1000000000.0);
			if(s->sortingDepth > maxSortingDepth) maxSortingDepth = s->sortingDepth;
			avgSortingDepth += s->sortingDepth;
		}
		
		sem_wait(&empty);
		sem_wait(&mutex);	
		buff[++producerCount] = i;
		sem_post(&mutex);
		sem_post(&full);
	}
	int divFactor = global_strm->state_fill_count;
	printf("[Time BWT] %lf\n",thread1_work);
	printf("[Average Time BWT] %lf\n",thread1_work/divFactor);
	printf("[Max Sorting Depth] %d\n",maxSortingDepth);
	printf("[Average Sorting Depth] %d\n",(int)((1.0*avgSortingDepth)/divFactor));
}

void *consume(void *arg) { 
	int item, i;
	bool is_last_block = false;
	
	struct timespec t1, t2;
	struct timespec tmerge; 
	double tmergeTotal = 0.0;
	double thread2_work = 0.0;
	
	for(i = 0; i <= global_strm->state_fill_count; i++) { 
	
		sem_wait(&full);
		sem_wait(&mutex);
		item = buff[++consumerCount];
		sem_post(&mutex);
		sem_post(&empty);
		
		clock_gettime(CLOCK_MONOTONIC, &t1);
		EState *s = (EState*)global_strm->state[item];
		merge_two_sort_arrays( s );
		clock_gettime(CLOCK_MONOTONIC, &tmerge);

		
		if(item == global_strm->state_fill_count) { 
			is_last_block = true;
		} else { 
			tmergeTotal += tmerge.tv_sec - t1.tv_sec + ((tmerge.tv_nsec - t1.tv_nsec)/1000000000.0);
		}

		if(!mtf_huff_wrapper( s, is_last_block )) { 
			printf("[ERROR] mtf_huff_wrapper\n");
			exit(1);
		}
		clock_gettime(CLOCK_MONOTONIC, &t2);
		thread2_work += t2.tv_sec - t1.tv_sec + ((t2.tv_nsec - t1.tv_nsec)/1000000000.0); 
	}
	int divFactor = global_strm->state_fill_count;
	printf("[Total Time Merge] %lf\n",tmergeTotal);
	printf("[Average Time Merge] %lf\n",tmergeTotal/divFactor);
	printf("[Time Merge, MTF+HUFF] %lf\n",thread2_work);
}

void BZ2_compressBlocks_pthreads ( bz_stream* strm)
{

	printf("[BZ2_compressBlocks_pthreads] Total Blocks : %d, Block Size %d\n", strm->state_fill_count, ((EState*)strm->state[0])->nblockMAX);
	gpuSetDevice(0);
	global_strm = strm;
	
	struct timespec t1, t2; 
	clock_gettime(CLOCK_MONOTONIC, &t1);
	
	pthread_t tid1, tid2;
	sem_init(&mutex, 0, 1);
	sem_init(&full, 0, 0);
	sem_init(&empty, 0, strm->state_fill_count);
	
	pthread_create(&tid1, NULL, produce, NULL);
	pthread_create(&tid2, NULL, consume, NULL);

	pthread_join(tid1, NULL);
	pthread_join(tid2, NULL);
	
	clock_gettime(CLOCK_MONOTONIC, &t2);
	double t = t2.tv_sec - t1.tv_sec + ((t2.tv_nsec - t1.tv_nsec)/1000000000.0); 
	printf("total compression time (with pthreads overlap) %lf\n",t);
}



void BZ2_compressBlocks_without_overlap (bz_stream* strm)  { 

	struct timespec t1, t2; 
	clock_gettime(CLOCK_MONOTONIC, &t1);

	UInt32 count;
	EState *s; 
	Bool is_last_block = False;

	for(count = 0; count <= strm->state_fill_count; count++) { 
		s = (EState *)strm->state[count];
		BZ_INITIALISE_CRC( s->blockCRC );
		if(count == strm->state_fill_count) { 
			is_last_block = True;
		}
		if (s->nblock > 0) {

			BZ_FINALISE_CRC ( s->blockCRC );
			s->combinedCRC = (s->combinedCRC << 1) | (s->combinedCRC >> 31);
			s->combinedCRC ^= s->blockCRC;
			if (s->blockNo > 1) s->numZ = 0;

			if (s->verbosity >= 2)
				VPrintf4( "    block %d: crc = 0x%08x, "
						"combined CRC = 0x%08x, size = %d\n",
						s->blockNo, s->blockCRC, s->combinedCRC, s->nblock );

			s->first_sort_length = gpuBlockSort( (UChar*) s->arr2, s->arr1, s->arr1_first_sort, s->arr1_second_sort, s->arr1_first_sort_rank, s->nblock, &(s->sortingDepth));

			merge_two_sort_arrays( s );
			
			UInt32* ptr    = s->ptr; 
			s->origPtr = -1;
			Int32 i;
			for (i = 0; i < s->nblock; i++)
				if (ptr[i] == 0)
				{ s->origPtr = i; break; };

			AssertH( s->origPtr != -1, 1003 );
		}
		s->zbits = (UChar*) (&((UChar*)s->arr2)[s->nblock]);

#ifdef PRINT_DEBUG
		printf("BZ2_compressBlock %d\n",s->blockNo); 
#endif

		// start with a fresh buffer with every block
		BZ2_bsInitWrite( s ); 

		/*-- If this is the first block, create the stream header. --*/
		if (s->blockNo == 1) {
#ifdef PRINT_DEBUG
			printf("This is the first block\n");
#endif
			//BZ2_bsInitWrite ( s );
			bsPutUChar ( s, BZ_HDR_B );
			bsPutUChar ( s, BZ_HDR_Z );
			bsPutUChar ( s, BZ_HDR_h );
			bsPutUChar ( s, (UChar)(BZ_HDR_0 + s->blockSize100k) );
		}

		if (s->nblock > 0) {

			bsPutUChar ( s, 0x31 ); bsPutUChar ( s, 0x41 );
			bsPutUChar ( s, 0x59 ); bsPutUChar ( s, 0x26 );
			bsPutUChar ( s, 0x53 ); bsPutUChar ( s, 0x59 );

			/*-- Now the block's CRC, so it is in a known place. --*/
			bsPutUInt32 ( s, s->blockCRC );

			/*-- 
			  Now a single bit indicating (non-)randomisation. 
			  As of version 0.9.5, we use a better sorting algorithm
			  which makes randomisation unnecessary.  So always set
			  the randomised bit to 'no'.  Of course, the decoder
			  still needs to be able to handle randomised blocks
			  so as to maintain backwards compatibility with
			  older versions of bzip2.
			  --*/
			bsW(s,1,0);

			bsW ( s, 24, s->origPtr );
			generateMTFValues ( s );
			sendMTFValues ( s );
		}


		/*-- If this is the last block, add the stream trailer. --*/
		if (is_last_block) {
#ifdef PRINT_DEBUG
			printf("This is the last block\n");
#endif
			bsPutUChar ( s, 0x17 ); bsPutUChar ( s, 0x72 );
			bsPutUChar ( s, 0x45 ); bsPutUChar ( s, 0x38 );
			bsPutUChar ( s, 0x50 ); bsPutUChar ( s, 0x90 );
			bsPutUInt32 ( s, s->combinedCRC );
			if (s->verbosity >= 2)
				VPrintf1( "    final combined CRC = 0x%08x\n   ", s->combinedCRC );
			bsFinishWrite ( s );
		}
	}
	
	clock_gettime(CLOCK_MONOTONIC, &t2);
	double t = t2.tv_sec - t1.tv_sec + ((t2.tv_nsec - t1.tv_nsec)/1000000000.0); 
	printf("total compression time (without overlap) %lf\n",t);
}
/*-------------------------------------------------------------*/
/*--- end                                        compress.c ---*/
/*-------------------------------------------------------------*/

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*   File:         seq_main.c   (an sequential version)                      */
/*   Description:  This program shows an example on how to call a subroutine */
/*                 that implements a simple k-means clustering algorithm     */
/*                 based on Euclid distance.                                 */
/*   Input file format:                                                      */
/*                 ascii  file: each line contains 1 data object             */
/*                 binary file: first 4-byte integer is the number of data   */
/*                 objects and 2nd integer is the no. of features (or        */
/*                 coordinates) of each object                               */
/*                                                                           */
/*   Author:  Wei-keng Liao                                                  */
/*            ECE Department Northwestern University                         */
/*            email: wkliao@ece.northwestern.edu                             */
/*   Copyright, 2005, Wei-keng Liao                                          */
/*                                                                           */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

// Copyright (c) 2005 Wei-keng Liao
// Copyright (c) 2011 Serban Giuroiu
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

// -----------------------------------------------------------------------------

#include <stdio.h>
#include <stdlib.h>
#include <string.h>     /* strlen() */
#include <unistd.h>     /* getopt */
#include <sys/stat.h>


void encode(char *input, int length, char *output);
int parseArguement(int argc, char **argv, char *inputFileName);
int readFile(char *inputFileName, int *inputSize, char **inputArr);


/*---< main() >-------------------------------------------------------------*/
int main(int argc, char **argv) {


    char inputFileName[64];
//    char input[] ="Chan YO YO YO YO YO YO YO YO";
//    int input_len = strlen(input);
    int inputSize = 0;
    char *inputArr;
    char *output;


    parseArguement(argc, argv, inputFileName);
    readFile(inputFileName, &inputSize, &inputArr);
    output = (char *)malloc(sizeof(char) * (inputSize + 1));
    printf("Before encoding\n");
    encode(inputArr,inputSize,output);
    printf("After encoding\n");

    //We should have compressed output in outptu buffer now
return 0;

}

int readFile(char *inputFileName, int *inputSize, char **inputArr ) {
    struct stat st;
    int i;
    char c;
    FILE *fp;
    stat(inputFileName, &st);
    *inputSize = st.st_size;
    //printf("size of the input file, %s: %d\n", inputFileName, inputSize);
    // Open file and handle error
    fp = fopen(inputFileName, "rb");
    if (fp == NULL){
        fprintf(stderr, "Cannot open file: %s\n", inputFileName);
        exit (-1);
    }
    // Allocate appropriate amount of char array and copy from file.
    //TODO: cuda malloc?
    *inputArr = (char *)malloc(*inputSize + 1);
    for (i = 0; (c = getc(fp)) != EOF; i++ ) {
        (*inputArr)[i] = c;
    } 
    (*inputArr)[i] = '\0';    

    return 0;
}

int parseArguement(int argc, char **argv, char *inputFileName) {
    int is_file = 0;
    char c;
    while( (c = getopt(argc, argv, "f:")) != -1) {
        switch(c) {
            case 'f':
                strcpy (inputFileName, optarg);
                is_file = 1;
                break;
            default:
                printf("Usage ./main -f FILENAME\n");
                exit(1);
        }
    }

    if((is_file == 0)){
        printf("Usage ./main -f FILENAME\n");
        exit(1);
    }
    return 1;
}













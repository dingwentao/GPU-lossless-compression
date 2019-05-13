CUDA_Compression_tool
=====================

 To compile, simply 
 $ make
 Usage for compression: ./main -i {inputfile} -o {outputfile}
 
 Usage for decompression: ./main -d 1 -i {inputfile} -o {outputfile}
 
 ---
 Default buffer size for the GPU pipeline is set to at the top of the main.c file. 
 	  Change BUFSIZE macro to change that variable.
 MINSIZE defines the minimum size required to make use of the GPU processing. Setting this
 	  variable to a lower value can lead program to indefinite behaviour. 
 
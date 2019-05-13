main: main.c culzss  gpu_compress deculzss  gpu_decompress decompression
	nvcc -g -L /usr/local/cuda/lib64/ -lcudart -lpthread -o main main.c culzss.o gpu_compress.o deculzss.o gpu_decompress.o decompression.o

decompression: 	decompression.c decompression.h
	nvcc -g  -c -lpthread -o decompression.o decompression.c
	
culzss:  culzss.c culzss.h 
	nvcc -g  -c -lpthread -o culzss.o culzss.c 

gpu_compress: gpu_compress.cu gpu_compress.h
	nvcc -O3 -g -c -lpthread -o gpu_compress.o gpu_compress.cu       

deculzss:  deculzss.c deculzss.h 
	nvcc -g  -c -lpthread -o deculzss.o deculzss.c

gpu_decompress: gpu_decompress.cu gpu_decompress.h
	nvcc -c -g  -lpthread -o gpu_decompress.o gpu_decompress.cu      
	
clean:
	rm *.o


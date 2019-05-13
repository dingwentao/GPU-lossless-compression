CC=nvcc

simpleCUDPP: scan_gold.o simpleCUDPP.o
	$(CC) -o simpleCUDPP scan_gold.o simpleCUDPP.o -L/home/dtao/GpuLossless/cudpp/lib -lcudpp -lm

scan_gold.o: scan_gold.cpp
	$(CC) -c scan_gold.cpp -I/home/dtao/GpuLossless/cudpp/include

simpleCUDPP.o: simpleCUDPP.cu
	$(CC) -c simpleCUDPP.cu -I/home/dtao/GpuLossless/cudpp/include


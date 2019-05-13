CUDA_INCLUDE = /usr/local/cuda/include
CUDA_LIB = /usr/local/cuda/lib64
ARCH = 70

CC = g++
NVCC = nvcc
AR = ar
CFLAGS = -std=c++17 -O3 -march=native -Wall -Werror
NVCC_FLAGS = --std=c++11 -O3 -Wno-deprecated-gpu-targets $(NVCC_ARCH)
RM_FLAGS = -f

INC_DIR = include
SRC_DIR = src
OBJ_DIR = bin
LIB_DIR = lib
EXEC_NAME = demo
LIB_NAME = cuhd

ENCODER = encoder
INC_ENCODER = $(ENCODER)/include
LIB_ENCODER = $(ENCODER)/bin

NVCC_ARCH = -gencode arch=compute_$(ARCH),code=sm_$(ARCH)
CUDA_LINK = -lcudart -lcuda

SRC_FILES := $(wildcard $(SRC_DIR)/*.cc)
CU_SRC_FILES := $(wildcard $(SRC_DIR)/*.cu)
OBJ_FILES := $(patsubst $(SRC_DIR)/%.cc,$(OBJ_DIR)/%.o,$(SRC_FILES))
CU_OBJ_FILES := $(patsubst $(SRC_DIR)/%.cu,$(OBJ_DIR)/%.cuo,$(CU_SRC_FILES))
ENCODER_OBJ_FILES := $(LIB_ENCODER)/*.o

default: link

cuhd: $(OBJ_FILES)

gpu: $(CU_OBJ_FILES)

lib: encoder cuhd gpu
	$(AR) rvs $(LIB_DIR)/$(LIB_NAME).a $(OBJ_FILES) $(CU_OBJ_FILES) $(ENCODER_OBJ_FILES)

.PHONY: encoder
encoder:
	$(MAKE) -C $(ENCODER)

link: encoder cuhd gpu
	$(CC) $(OBJ_FILES) $(CU_OBJ_FILES) $(ENCODER_OBJ_FILES) -L $(CUDA_LIB) -o $(OBJ_DIR)/$(EXEC_NAME) $(CUDA_LINK)

$(OBJ_DIR)/%.cuo: $(SRC_DIR)/%.cu
	$(NVCC) $(NVCC_FLAGS) -I $(INC_DIR) -c -o $@ $<

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cc
	$(CC) $(CFLAGS) -I $(INC_DIR) -I $(INC_ENCODER) -I $(CUDA_INCLUDE) -c -o $@ $<

.PHONY: clean
clean: clean_encoder
	rm $(RM_FLAGS) $(OBJ_DIR)/*.o
	rm $(RM_FLAGS) $(OBJ_DIR)/*.cuo
	rm $(RM_FLAGS) $(OBJ_DIR)/$(EXEC_NAME)
	rm $(RM_FLAGS) $(LIB_DIR)/$(LIB_NAME).a
	rm $(RM_FLAGS) $(LIB_DIR)/$(LIB_NAME).so

.PHONY: clean_encoder
clean_encoder:
	$(MAKE) -C encoder clean

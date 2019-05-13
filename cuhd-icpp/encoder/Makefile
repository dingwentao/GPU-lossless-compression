CFLAGS = -std=c++17 -O3 -march=native -Wall -Werror
RM_FLAGS = -f

INC_DIR = include
INC_DECODER = ../include
SRC_DIR = src
SRC_DIR_DECODER = ../src
OBJ_DIR = bin
OBJ_DIR_DECODER = ../bin
EXEC_NAME = main

SRC_FILES := $(wildcard $(SRC_DIR)/*.cc)
SRC_FILES_DECODER := $(SRC_DIR_DECODER)/cuhd_codetable.cc
OBJ_FILES := $(patsubst $(SRC_DIR)/%.cc,$(OBJ_DIR)/%.o,$(SRC_FILES))
OBJ_FILES_DECODER := $(patsubst $(SRC_DIR_DECODER)/%.cc,$(OBJ_DIR_DECODER)/%.o,$(SRC_FILES_DECODER))

default: llhc

llhc: $(OBJ_FILES)

cuhd: $(OBJ_FILES_DECODER)

link: cuhd llhc
	g++ $(OBJ_FILES) $(OBJ_FILES_DECODER) -o $(OBJ_DIR)/$(EXEC_NAME)

$(OBJ_DIR_DECODER)/%.o: $(SRC_DIR_DECODER)/%.cc
	g++ $(CFLAGS) -I $(INC_DECODER) -c -o $@ $<

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cc
	g++ $(CFLAGS) -I $(INC_DIR) -I $(INC_DECODER) -c -o $@ $<

.PHONY: clean
clean:
	rm $(RM_FLAGS) $(OBJ_FILES)
	rm $(RM_FLAGS) $(OBJ_FILES_DECODER)
	rm $(RM_FLAGS) $(OBJ_DIR)/$(EXEC_NAME)

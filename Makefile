CC = nvcc
OPT_FLAG = -O3
OPT_FLAG_CUDA =
WARNINGS = -Wall -Wextra -Wshadow
CFLAGS = -std=c++20 -arch=native $(OPT_FLAG_CUDA) -dlto -Iinclude -Xcompiler "$(OPT_FLAG) -march=native -fopenmp $(WARNINGS)" -MMD -MP -g
LDLIBS = -lm -lgomp

SRC_DIR = src
BUILD_DIR = build
BIN_DIR = bin
INPUT_DIR = input
OUTPUT_DIR = output
OUTPUT_IMAGE = output-image

TARGET = $(BIN_DIR)/exa_beam

SRCS_CPP = $(wildcard $(SRC_DIR)/*.cpp)
SRCS_CU  = $(wildcard $(SRC_DIR)/*.cu)

OBJS_CPP = $(patsubst $(SRC_DIR)/%.cpp, $(BUILD_DIR)/%.o, $(SRCS_CPP))
OBJS_CU  = $(patsubst $(SRC_DIR)/%.cu, $(BUILD_DIR)/%.o, $(SRCS_CU))

OBJS = $(OBJS_CPP) $(OBJS_CU)

all: output-dirs $(TARGET)

fast: OPT_FLAG = -Ofast
fast: OPT_FLAG_CUDA = -use_fast_math
fast: all

$(TARGET): $(OBJS) | $(BIN_DIR)
	@$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)
	$(info Linked $@.)

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp | $(BUILD_DIR)
	@$(CC) $(CFLAGS) -c $< -o $@
	$(info Compiled CPU object $@.)

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cu | $(BUILD_DIR)
	@$(CC) $(CFLAGS) -c $< -o $@
	$(info Compiled CUDA object $@.)

$(BIN_DIR) $(BUILD_DIR):
	@mkdir -p $@
	$(info Created $@ directory.)

output-dirs:
	@mkdir -p $(OUTPUT_DIR) $(OUTPUT_IMAGE) $(INPUT_DIR)
	$(info Created output directories.)

clean:
	@rm -rf $(BUILD_DIR) $(BIN_DIR) $(OUTPUT_DIR) $(OUTPUT_IMAGE) $(INPUT_DIR)
	$(info Removed output directories.)

-include $(OBJS:.o=.d)

.PHONY: all clean output-dirs fast
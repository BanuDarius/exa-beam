CC = g++
OPT_FLAG = -O3
WARNINGS = -Wall -Wextra -Wshadow
CFLAGS = -std=c++20 $(OPT_FLAG) -march=native -Iinclude -fopenmp -flto -MMD -MP -g $(WARNINGS)
LDLIBS = -lm

SRC_DIR = src
BUILD_DIR = build
BIN_DIR = bin
INPUT_DIR = input
OUTPUT_DIR = output
OUTPUT_IMAGE = output-image

TARGET = $(BIN_DIR)/exa_beam

SRCS = $(wildcard $(SRC_DIR)/*.cpp)

OBJS = $(patsubst $(SRC_DIR)/%.cpp, $(BUILD_DIR)/%.o, $(SRCS))

all: output-dirs $(TARGET)

fast: OPT_FLAG = -Ofast
fast: all

$(TARGET): $(OBJS) | $(BIN_DIR)
	@$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)
	$(info Linked $@.)

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp | $(BUILD_DIR)
	@$(CC) $(CFLAGS) -c $< -o $@
	$(info Compiled $@.)

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

.PHONY: all clean output_dirs fast
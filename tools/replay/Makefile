# Makefile for fake_gemm (W4A8 per-group kernel replay)

# ====== toolchain / arch ======
SM        ?= 89
NVCC      ?= nvcc
PTXAS     ?= ptxas
CUOBJDUMP ?= cuobjdump

# ====== targets / sources ======
TARGET       := fake_gemm
TARGET_DEBUG := $(TARGET)_debug

SRCS        := fake_launch_gemm_w4a8_per_group.cu
KERNEL_SRC  :=
HEADERS     :=

# ====== common flags ======

# double GENCODE, 同时嵌入 SASS 和 PTX
GENCODE := \
  -gencode arch=compute_$(SM),code=sm_$(SM) \
  -gencode arch=compute_$(SM),code=compute_$(SM)

GENCODE_DEBUG := -gencode arch=compute_$(SM),code=sm_$(SM)

NVCCFLAGS_RELEASE := -std=c++17 -lineinfo -Xptxas -O3,-v
NVCCFLAGS_DEBUG   := -std=c++17 -G -O0 -g -lineinfo -Xptxas -O0,-v -DDEBUG -Xcompiler -fno-omit-frame-pointer

# ptxas 参数（和 -Xptxas 对应）
PTXASFLAGS := -O3 -v --warn-on-spills --generate-line-info

.PHONY: all build debug run run-debug ptx sass clean

# ====== default ======
all: build

# ====== builds ======
build: $(TARGET)

debug: $(TARGET_DEBUG)

$(TARGET): $(SRCS) $(HEADERS)
	$(NVCC) $(GENCODE) $(NVCCFLAGS_RELEASE) -o $@ $(SRCS)

$(TARGET_DEBUG): $(SRCS) $(HEADERS)
	$(NVCC) $(GENCODE_DEBUG) $(NVCCFLAGS_DEBUG) -o $@ $(SRCS)

# ====== run ======
run: build
	./$(TARGET)

run-debug: debug
	./$(TARGET_DEBUG)

# ====== 直接从最终可执行文件里反汇编（与最终链接保持一致） ======
sass: build
	$(CUOBJDUMP) --dump-sass $(TARGET) > $(TARGET).sass
	@echo "SASS (from executable) written to $(TARGET).sass"

ptx:
	$(CUOBJDUMP) --dump-ptx  $(TARGET) > $(TARGET).ptx
	@echo "PTX  (from executable) written to $(TARGET).ptx"

# ====== clean ======
clean:
	rm -f $(TARGET) $(TARGET_DEBUG) \
	      $(TARGET).ptx $(TARGET).cubin $(TARGET).sass $(TARGET).sass

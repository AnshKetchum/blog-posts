#!/bin/bash
set -e

# ----------------------------
# Clean and create build folders
# ----------------------------
rm -rf build_optimized build_unoptimized
mkdir -p build_optimized build_unoptimized

# ----------------------------
# Compile OPTIMIZED version (register accumulation)
# ----------------------------
echo "Building OPTIMIZED (register accumulation)..."
rm -f main.o solution.o
nvcc -G \
     -arch=sm_89 \
     -O3 \
     -Xptxas=-v \
     --resource-usage \
     --keep \
     --keep-dir=build_optimized \
     -lineinfo \
     src/main.cu src/solution_optimized.cu \
     -o build_optimized/matrix_mult

# Dump SASS and PTX for optimized
cuobjdump -sass build_optimized/solution_optimized.sm_89.cubin > build_optimized/solution_optimized.sass
cuobjdump -ptx build_optimized/solution_optimized.sm_89.cubin > build_optimized/solution_optimized.ptx

echo "Optimized register usage:"
grep -A 5 "matrix_multiplication_kernel" build_optimized/solution_optimized.sass | head -10

# ----------------------------
# Compile UNOPTIMIZED version (volatile, memory traffic)
# ----------------------------
echo -e "\nBuilding UNOPTIMIZED (volatile, memory traffic)..."
rm -f main.o solution.o
nvcc -G \
     -arch=sm_89 \
     -O0 \
     -Xptxas=-O0 \
     -Xcompiler=-O0 \
     -Xptxas=-v \
     --resource-usage \
     --keep \
     --keep-dir=build_unoptimized \
     -lineinfo \
     src/main.cu src/solution_unoptimized.cu \
     -o build_unoptimized/matrix_mult

# Dump SASS and PTX for unoptimized
cuobjdump -sass build_unoptimized/solution_unoptimized.sm_89.cubin > build_unoptimized/solution_unoptimized.sass
cuobjdump -ptx build_unoptimized/solution_unoptimized.sm_89.cubin > build_unoptimized/solution_unoptimized.ptx

echo "Unoptimized register usage:"
grep -A 5 "matrix_multiplication_kernel" build_unoptimized/solution_unoptimized.sass | head -10

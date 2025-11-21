#include <cuda_runtime.h>
#include "stdio.h"

__global__ void matrix_multiplication_kernel(const float* A, const float* B, volatile float* C, int M, int N, int K) {
    int y = blockIdx.x * blockDim.x + threadIdx.x;
    int x = blockIdx.y * blockDim.y + threadIdx.y;

    
    if(x < M && y < K) {
        // printf("Thread indices entering if - %d %d\n", x, y);
        for(int i = 0; i < N; i++) {
            C[x * K + y] += A[x * N + i] * B[i * K + y];    
        }
    }
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((K + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (M + threadsPerBlock.y - 1) / threadsPerBlock.y);
    
    matrix_multiplication_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}

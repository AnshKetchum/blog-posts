#include <iostream>
#include <fstream>
#include <vector>
#include <cuda_runtime.h>

extern "C" void solve(const float* A, const float* B, float* C, int M, int N, int K);

// Dummy kernel to flush memory
__global__ void flush_kernel(float* buf, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        buf[idx] = buf[idx] * 1.000001f;
    }
}

bool readMatrix(const std::string& filename, std::vector<float>& mat, int& rows, int& cols) {
    std::ifstream fin(filename);
    if (!fin.is_open()) {
        std::cerr << "Error: Cannot open " << filename << std::endl;
        return false;
    }
    fin >> rows >> cols;
    mat.resize(rows * cols);
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c)
            fin >> mat[r * cols + c];
    return true;
}

void writeMatrix(const std::string& filename, const float* mat, int rows, int cols) {
    std::ofstream fout(filename);
    fout << rows << " " << cols << "\n";
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c)
            fout << mat[r * cols + c] << " ";
        fout << "\n";
    }
}

int main() {
    std::vector<float> h_A, h_B;
    int M, N, N2, K;

    // -----------------------------
    // Read input matrices
    // -----------------------------
    if (!readMatrix("A.txt", h_A, M, N)) return 1;
    if (!readMatrix("B.txt", h_B, N2, K)) return 1;

    if (N != N2) {
        std::cerr << "Error: A columns (" << N << ") != B rows (" << N2 << ")\n";
        return 1;
    }

    std::cout << "Matrix A: " << M << " x " << N << "\n";
    std::cout << "Matrix B: " << N2 << " x " << K << "\n";

    cudaDeviceReset();

    // -----------------------------
    // Allocate device matrices
    // -----------------------------
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, M * N * sizeof(float));
    cudaMalloc((void**)&d_B, N * K * sizeof(float));
    cudaMalloc((void**)&d_C, M * K * sizeof(float));

    cudaMemcpy(d_A, h_A.data(), M * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), N * K * sizeof(float), cudaMemcpyHostToDevice);

    // -----------------------------
    // Allocate flush buffer
    // -----------------------------
    size_t flushSize = M*N + N*K + M*K;
    float* d_flush;
    cudaMalloc((void**)&d_flush, flushSize * sizeof(float));
    std::cout << "Flush buffer allocated: " << flushSize << " floats\n";

    // -----------------------------
    // Disable L1 cache for solve()
    // -----------------------------
    cudaFuncSetCacheConfig(solve, cudaFuncCachePreferNone);
    std::cout << "L1 cache disabled for solve()\n";

    const int numFlushTrials = 3;
    const int numBenchmarkTrials = 10;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // -----------------------------
    // Flush caches several times before benchmark
    // -----------------------------
    std::cout << "Performing " << numFlushTrials << " cache flush trials...\n";
    for (int i = 0; i < numFlushTrials; i++) {
        flush_kernel<<<(flushSize + 255)/256, 256>>>(d_flush, flushSize);
        cudaDeviceSynchronize();
        std::cout << "  Flush trial " << (i+1) << " complete\n";
    }

    // -----------------------------
    // Warmup
    // -----------------------------
    std::cout << "Running warmup...\n";
    solve(d_A, d_B, d_C, M, N, K);
    cudaDeviceSynchronize();
    std::cout << "Warmup complete.\n";

    // -----------------------------
    // Benchmark
    // -----------------------------
    float totalTimeMs = 0.0f;
    for (int t = 0; t < numBenchmarkTrials; t++) {
        // flush_kernel<<<(flushSize + 255)/256, 256>>>(d_flush, flushSize);
        // cudaDeviceSynchronize();

        cudaEventRecord(start);
        solve(d_A, d_B, d_C, M, N, K);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        totalTimeMs += ms;

        if ((t+1) % 50 == 0) { // Print progress every 50 trials
            std::cout << "  Trial " << (t+1) << "/" << numBenchmarkTrials << ": " << ms << " ms\n";
        }
    }

    float avgTimeMs = totalTimeMs / numBenchmarkTrials;
    std::cout << "Average matmul kernel runtime over " << numBenchmarkTrials 
              << " trials: " << avgTimeMs << " ms" << std::endl;

    // -----------------------------
    // Copy result back to host
    // -----------------------------
    std::vector<float> h_C(M * K);
    cudaMemcpy(h_C.data(), d_C, M * K * sizeof(float), cudaMemcpyDeviceToHost);
    writeMatrix("C.txt", h_C.data(), M, K);
    std::cout << "Output written to C.txt\n";

    // -----------------------------
    // Cleanup
    // -----------------------------
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_flush);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    std::cout << "All done.\n";
    return 0;
}

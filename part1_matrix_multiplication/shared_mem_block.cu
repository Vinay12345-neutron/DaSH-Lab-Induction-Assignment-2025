//==========================================================================
// shared_mem_block.cu – GEMM with shared‑memory blocking
//==========================================================================
#ifndef uint
#define uint unsigned int
#endif


#include <iostream>
#include <random>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdint>          // for uint32_t

// ---------------------------------------------------------------------
// Simple error‑checking macro
// ---------------------------------------------------------------------
#define CUDA_CHECK(err)                                                   \
    do {                                                                  \
        cudaError_t e = (err);                                            \
        if (e != cudaSuccess) {                                           \
            std::cerr << "CUDA error " << __FILE__ << ':' << __LINE__      \
                      << " : " << cudaGetErrorString(e) << std::endl;      \
            std::exit(EXIT_FAILURE);                                      \
        }                                                                 \
    } while (0)

// ---------------------------------------------------------------------
// Integer ceiling division (used for grid sizing)
// ---------------------------------------------------------------------
#define CEIL_DIV(x, y)   ( ((x) + (y) - 1) / (y) )

// ---------------------------------------------------------------------
// GEMM kernel – shared‑memory blocking
// ---------------------------------------------------------------------
// BLOCKSIZE is the tile dimension (e.g. 32).  The block is 1‑D with
// BLOCKSIZE*BLOCKSIZE threads.  Each thread computes one element of C.
template <uint32_t BLOCKSIZE>
__global__ void shared_mem_block(int M, int N, int K,
                                 float alpha,
                                 const float* A,
                                 const float* B,
                                 float beta,
                                 float* C)
{
    const uint cRow = blockIdx.x;                     // tile row
    const uint cCol = blockIdx.y;                     // tile col

    __shared__ float As[BLOCKSIZE * BLOCKSIZE];
    __shared__ float Bs[BLOCKSIZE * BLOCKSIZE];

    const uint threadCol = threadIdx.x % BLOCKSIZE; // coordinates of thread inside the tile that the block is working on
    const uint threadRow = threadIdx.x / BLOCKSIZE;

    // advance pointers to the first element of this tile
    A += cRow * BLOCKSIZE * K;                       // row = cRow, col = 0
    B += cCol * BLOCKSIZE;                           // row = 0,   col = cCol
    C += cRow * BLOCKSIZE * N + cCol * BLOCKSIZE;    // row = cRow, col = cCol

    float tmp = 0.0f;

    for (int bkIdx = 0; bkIdx < K; bkIdx += BLOCKSIZE) {
        // load one tile of A and B into shared memory (coalesced)
        As[threadRow * BLOCKSIZE + threadCol] = A[threadRow * K + threadCol];
        Bs[threadRow * BLOCKSIZE + threadCol] = B[threadRow * N + threadCol];
        __syncthreads();

        // compute partial dot‑product using the shared tiles
        for (int dotIdx = 0; dotIdx < BLOCKSIZE; ++dotIdx) {
            tmp += As[threadRow * BLOCKSIZE + dotIdx] *
                   Bs[dotIdx * BLOCKSIZE + threadCol];
        }
        __syncthreads();

        // move to the next K‑chunk
        A += BLOCKSIZE;
        B += BLOCKSIZE * N;
    }

    C[threadRow * N + threadCol] = alpha * tmp + beta * C[threadRow * N + threadCol];
}

// ---------------------------------------------------------------------
// Helper that mirrors the “run_global_mem_coalesce” style
// ---------------------------------------------------------------------
void run_shared_mem_block(int M, int N, int K,
                         float alpha, float* dA, float* dB,
                         float beta,  float* dC)
{
    constexpr uint32_t TILE = 32;                     // 32×32 = 1024 threads
    dim3 grid( CEIL_DIV(M, TILE), CEIL_DIV(N, TILE) );
    dim3 block( TILE * TILE );                        // 1‑D block

    // carve out as much L1 as possible for shared memory (optional)
    cudaFuncSetAttribute(
        shared_mem_block<TILE>,
        cudaFuncAttributePreferredSharedMemoryCarveout,
        cudaSharedmemCarveoutMaxShared);

    shared_mem_block<TILE><<<grid, block>>>(M, N, K,
                                           alpha, dA, dB,
                                           beta,  dC);
}

// ---------------------------------------------------------------------
// Fill a buffer with deterministic pseudo‑random numbers [0,1)
// ---------------------------------------------------------------------
static void random_fill(float* p, size_t n)
{
    std::mt19937 rng(0);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (size_t i = 0; i < n; ++i) p[i] = dist(rng);
}

// ---------------------------------------------------------------------
// GFLOPS = 2·M·N·K / (seconds·1e9)
// ---------------------------------------------------------------------
static double gflops(int M, int N, int K, double sec)
{
    return 2.0 * M * N * K / (sec * 1e9);
}

// ---------------------------------------------------------------------
// Main driver
// ---------------------------------------------------------------------
int main()
{
    std::cout << "Running Shared‑Memory‑Block Implementation" << std::endl;

    const int maxSize = 4096;
    const int sizes[] = {128, 256, 512, 1024, 2048, 4096};
    const int nSizes   = sizeof(sizes) / sizeof(sizes[0]);

    // --------------------------------------------------------------
    // Allocate host buffers for the *maximum* matrix size
    // --------------------------------------------------------------
    size_t maxElems = static_cast<size_t>(maxSize) * maxSize;
    float *hA = (float*)malloc(maxElems * sizeof(float));
    float *hB = (float*)malloc(maxElems * sizeof(float));
    float *hC = (float*)malloc(maxElems * sizeof(float));
    if (!hA || !hB || !hC) {
        std::cerr << "Host allocation failed\n";
        return EXIT_FAILURE;
    }

    random_fill(hA, maxElems);
    random_fill(hB, maxElems);
    random_fill(hC, maxElems);

    // --------------------------------------------------------------
    // Allocate device buffers (same max size) and copy once
    // --------------------------------------------------------------
    float *dA, *dB, *dC;
    CUDA_CHECK(cudaMalloc(&dA, maxElems * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dB, maxElems * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dC, maxElems * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(dA, hA, maxElems * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB, maxElems * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dC, hC, maxElems * sizeof(float),
                          cudaMemcpyHostToDevice));

    // --------------------------------------------------------------
    // Benchmark parameters
    // --------------------------------------------------------------
    const float alpha = 0.5f;
    const float beta  = 3.0f;
    const int   repeat = 50;               // timed launches per size

    // --------------------------------------------------------------
    // Loop over the six matrix sizes
    // --------------------------------------------------------------
    for (int i = 0; i < nSizes; ++i) {
        int M = sizes[i];
        int N = sizes[i];
        int K = sizes[i];

        // Warm‑up launch (removes first‑run overhead)
        run_shared_mem_block(M, N, K, alpha, dA, dB, beta, dC);
        CUDA_CHECK(cudaDeviceSynchronize());

        // ----------------------------------------------------------
        // Timing with CUDA events
        // ----------------------------------------------------------
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        CUDA_CHECK(cudaEventRecord(start));
        for (int r = 0; r < repeat; ++r) {
            run_shared_mem_block(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= repeat;                     // average per launch (ms)

        double sec = ms * 1e-3;           // seconds for printing
        double perf = gflops(M, N, K, sec);

        // ----- EXACT output format -----
        printf("dimensions(m=n=k) %d, alpha: %.1f, beta: %.1f\n",
               M, alpha, beta);
        printf("Average elapsed time: (%.6f) s, performance: (%8.1f) GFLOPS. size: (%d).\n",
               sec, perf, M);
        // ------------------------------

        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
    }

    // --------------------------------------------------------------
    // Clean‑up
    // --------------------------------------------------------------
    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));
    free(hA);
    free(hB);
    free(hC);

    return 0;
}
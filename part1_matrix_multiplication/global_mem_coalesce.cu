//==========================================================================
// global_mem_coalesce.cu – GEMM with coalesced global loads
//==========================================================================

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
// GEMM kernel – global‑memory coalescing
// ---------------------------------------------------------------------
// BLOCKSIZE is the tile dimension (e.g. 16).  The block is 1‑D with
// BLOCKSIZE*BLOCKSIZE threads.  Each thread computes one C element.
template <uint32_t BLOCKSIZE>
__global__ void global_mem_coalesce(int M, int N, int K,
                                   float alpha,
                                   const float* A,
                                   const float* B,
                                   float beta,
                                   float* C)
{
    // 1‑D thread index → (row, col) inside the tile
    const int cRow = blockIdx.x * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);
    const int cCol = blockIdx.y * BLOCKSIZE + (threadIdx.x % BLOCKSIZE);

    if (cRow < M && cCol < N) {
        float acc = 0.0f;
        for (int i = 0; i < K; ++i) {
            // Row‑major A, column‑major B (same as naïve)
            acc += A[cRow * K + i] * B[i * N + cCol];
        }
        C[cRow * N + cCol] = alpha * acc + beta * C[cRow * N + cCol];
    }
}

void run_global_mem_coalesce(int M, int N, int K,
                             float alpha, float* dA, float* dB,
                             float beta,  float* dC)
{
    // Choose a tile size that matches the kernel template.
    // 32 → 32×32 = 1024 threads per block (max for modern GPUs).
    constexpr uint32_t TILE = 32;

    dim3 grid( CEIL_DIV(M, TILE), CEIL_DIV(N, TILE) );
    dim3 block( TILE * TILE );               // 1‑D block

    // Instantiate the templated kernel with BLOCKSIZE = TILE
    global_mem_coalesce<TILE><<<grid, block>>>(M, N, K,
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
    std::cout << "Running Global‑Memory‑Coalesce Implementation" << std::endl;

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
        run_global_mem_coalesce(M, N, K, alpha, dA, dB, beta, dC);
        CUDA_CHECK(cudaDeviceSynchronize());

        // ----------------------------------------------------------
        // Timing with CUDA events
        // ----------------------------------------------------------
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        CUDA_CHECK(cudaEventRecord(start));
        for (int r = 0; r < repeat; ++r) {
            run_global_mem_coalesce(M, N, K, alpha, dA, dB, beta, dC);
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
//==========================================================================
// 1d_block_tiling.cu – GEMM with 1‑D block tiling (register‑blocked)
//==========================================================================

#include <iostream>
#include <random>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdint>          // for uint32_t
#include <cassert>          // for assert

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
// GEMM kernel – 1‑D block tiling 
// ---------------------------------------------------------------------
template <int BM, int BN, int BK, int TM>
__global__ void sgemm1DBlocktiling(int M, int N, int K,
                                   float alpha,
                                   const float *A,
                                   const float *B,
                                   float beta,
                                   float *C)
{
    // Tile indices (each block works on a BM×BN tile of C)
    const unsigned int cRow = blockIdx.y;   // tile‑row
    const unsigned int cCol = blockIdx.x;   // tile‑col

    // Thread coordinates inside the tile
    const int threadCol = threadIdx.x % BN;   // 0 … BN‑1
    const int threadRow = threadIdx.x / BN;   // 0 … BM‑1

    // Shared‑memory buffers for the current A‑ and B‑tiles
    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    // Move the global pointers to the first element of this tile
    A += cRow * BM * K;                     // start of the tile’s rows in A
    B += cCol * BN;                         // start of the tile’s columns in B
    C += cRow * BM * N + cCol * BN;          // top‑left element of the output tile

    // Sanity checks – they fire if the launch configuration is wrong
    assert(BM * BK == blockDim.x);
    assert(BN * BK == blockDim.x);

    // Ind the shared‑memory tiles (warp‑level coalescing)
    const unsigned int innerColA = threadIdx.x % BK;
    const unsigned int innerRowA = threadIdx.x / BK;
    const unsigned int innerColB = threadIdx.x % BN;
    const unsigned int innerRowB = threadIdx.x / BN;

    // Register‑level accumulator (each thread produces TM rows)
    float threadResults[TM] = {0.0f};

    // -----------------------------------------------------------------
    // Loop over the K dimension in BK‑wide chunks
    // -----------------------------------------------------------------
    for (unsigned int bkIdx = 0; bkIdx < K; bkIdx += BK)
    {
        // Load one BM×BK tile of A and one BK×BN tile of B into shared memory
        As[innerRowA * BK + innerColA] = A[innerRowA * K + innerColA];
        Bs[innerRowB * BN + innerColB] = B[innerRowB * N + innerColB];
        __syncthreads();
        // __syncth Advance the global pointers to the next K-chunk
        A += BK;            // move right by BK columns in A
        B += BK * N;        // move down by BK rows in B

        // Compute the partial dot‑product for this chunk
        for (unsigned int dotIdx = 0; dotIdx < BK; ++dotIdx)
        {
            float tmpB = Bs[dotIdx * BN + threadCol];
            for (unsigned int resIdx = 0; resIdx < TM; ++resIdx)
            {
                threadResults[resIdx] +=
                    As[(threadRow * TM + resIdx) * BK + dotIdx] * tmpB;
            }
        }
        __syncthreads();
    }

    // -----------------------------------------------------------------
    // Write the final results back to global memory
    // -----------------------------------------------------------------
    for (unsigned int resIdx = 0; resIdx < TM; ++resIdx)
    {
        C[(threadRow * TM + resIdx) * N + threadCol] =
            alpha * threadResults[resIdx] +
            beta  * C[(threadRow * TM + resIdx) * N + threadCol];
    }
}

void run1DBlocktiling(int M, int N, int K,
                      float alpha,
                      float *A, float *B,
                      float beta, float *C)
{
    constexpr unsigned int BM = 64;
    constexpr unsigned int BN = 64;
    constexpr unsigned int BK = 8;
    constexpr unsigned int TM = 8;

    dim3 gridDim( CEIL_DIV(N, BN), CEIL_DIV(M, BM) );
    dim3 blockDim( (BM * BN) / TM );   // e.g. (64*64)/8 = 512 threads

    sgemm1DBlocktiling<BM, BN, BK, TM>
        <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
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
    std::cout << "Running 1D‑Block‑Tiling Implementation" << std::endl;

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
        run1DBlocktiling(M, N, K, alpha, dA, dB, beta, dC);
        CUDA_CHECK(cudaDeviceSynchronize());

        // ----------------------------------------------------------
        // Timing with CUDA events
        // ----------------------------------------------------------
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        CUDA_CHECK(cudaEventRecord(start));
        for (int r = 0; r < repeat; ++r) {
            run1DBlocktiling(M, N, K, alpha, dA, dB, beta, dC);
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
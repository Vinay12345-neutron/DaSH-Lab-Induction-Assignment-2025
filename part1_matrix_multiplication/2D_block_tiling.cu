//==========================================================================
// 2D_block_tiling.cu – GEMM with 2‑D block tiling (register‑blocked)
//==========================================================================
// for Windows adding below block of code
#ifndef uint
#define uint unsigned int
#endif
// 


#include <iostream>
#include <random>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdint>          // for uint32_t
#include <cassert>   
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))       // for assert

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
// Kernel – 2‑D block tiling
// ---------------------------------------------------------------------

template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void __launch_bounds__((BM * BN) / (TM * TN), 1)
sgemm2DBlocktiling(int M, int N, int K, float alpha, const float *A,
                   const float *B, float beta, float *C)
{
    // -----------------------------------------------------------------
    // 1) Identify the output tile this block will compute
    // -----------------------------------------------------------------
    const uint cRow = blockIdx.y;                     // tile‑row index
    const uint cCol = blockIdx.x;                     // tile‑col index
    const uint totalResultsBlocktile = BM * BN;        // #output elements per tile

    // A thread produces TM × TN results inside the tile
    const uint numThreadsBlocktile = totalResultsBlocktile / (TM * TN);
    // sanity check – the launch must provide exactly this many threads
    assert(numThreadsBlocktile == blockDim.x);

    // BN/TN threads span a column of the tile
    const int threadCol = threadIdx.x % (BN / TN);
    const int threadRow = threadIdx.x / (BN / TN);

    // -----------------------------------------------------------------
    // 2) Shared‑memory tiles for A and B
    // -----------------------------------------------------------------
    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    // Move the global pointers to the first element of this tile
    A += cRow * BM * K;                     // start of A‑tile rows
    B += cCol * BN;                         // start of B‑tile columns
    C += cRow * BM * N + cCol * BN;         // start of C‑tile

    // -----------------------------------------------------------------
    // 3) Indices used for loading the shared‑memory tiles
    // -----------------------------------------------------------------
    const uint innerRowA = threadIdx.x / BK;
    const uint innerColA = threadIdx.x % BK;
    const uint strideA   = numThreadsBlocktile / BK;   // rows of A loaded per step

    const uint innerRowB = threadIdx.x / BN;
    const uint innerColB = threadIdx.x % BN;
    const uint strideB   = numThreadsBlocktile / BN;   // rows of B loaded per step

    // -----------------------------------------------------------------
    // 4) Thread‑local accumulators (register‑blocked)
    // -----------------------------------------------------------------
    float threadResults[TM * TN] = {0.0f};   // final TM × TN results for this thread
    float regM[TM] = {0.0f};                // registers for a column of A
    float regN[TN] = {0.0f};                // registers for a row    of B

    // -----------------------------------------------------------------
    // 5) Main K‑loop – load tiles, compute, advance pointers
    // -----------------------------------------------------------------
    for (uint bkIdx = 0; bkIdx < K; bkIdx += BK)
    {
        // ---- load a BM×BK tile of A into shared memory -----------------
        for (uint loadOffset = 0; loadOffset < BM; loadOffset += strideA)
        {
            As[(innerRowA + loadOffset) * BK + innerColA] =
                A[(innerRowA + loadOffset) * K + innerColA];
        }

        // ---- load a BK×BN tile of B into shared memory -----------------
        for (uint loadOffset = 0; loadOffset < BK; loadOffset += strideB)
        {
            Bs[(innerRowB + loadOffset) * BN + innerColB] =
                B[(innerRowB + loadOffset) * N + innerColB];
        }

        __syncthreads();

        // ---- advance global pointers to the next K‑chunk ----------------
        A += BK;          // move right by BK columns in A
        B += BK * N;      // move down by BK rows in B

        // ---- compute the partial products for this K‑chunk ------------
        // 2‑D block tiling – each thread produces TM × TN results
        for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) // iterates over K-tiles
        {
            // load TM elements of the current column of A (one per row) into registers
            for (uint i = 0; i < TM; ++i)
                regM[i] = As[(threadRow * TM + i) * BK + dotIdx];
                // regM holds A[ row = threadRow*TM+i , col = dotIdx ]

            // load TN elements of the current row of B (one per column) into registers
            for (uint i = 0; i < TN; ++i)
                regN[i] = Bs[dotIdx * BN + threadCol * TN + i];
                // regN holds B[ row = dotIdx , col = threadCol*TN+i ]

            // outer‑product of the two vectors → TM×TN partial result, accumulate TM × TN products
            for (uint resIdxM = 0; resIdxM < TM; ++resIdxM)
            {
                for (uint resIdxN = 0; resIdxN < TN; ++resIdxN)
                {
                    threadResults[resIdxM * TN + resIdxN] +=
                        regM[resIdxM] * regN[resIdxN];
                        // each combination (row, col) of the mini‑matrix gets one product
                }
            }
        }

        __syncthreads();   // ensure all threads finished before next tile
    }

    // -----------------------------------------------------------------
    // 6) Write the TM × TN results back to global memory
    // -----------------------------------------------------------------
    for (uint resIdxM = 0; resIdxM < TM; ++resIdxM)
    {
        for (uint resIdxN = 0; resIdxN < TN; ++resIdxN)
        {
            uint globalRow = (threadRow * TM + resIdxM);
            uint globalCol = (threadCol * TN + resIdxN);
            C[globalRow * N + globalCol] =
                alpha * threadResults[resIdxM * TN + resIdxN] +
                beta  * C[globalRow * N + globalCol];
        }
    }
}

// ---------------------------------------------------------------------
// Launcher 
// ---------------------------------------------------------------------
void runSgemm2DBlocktiling(int M, int N, int K,
                           float alpha, float *A, float *B,
                           float beta, float *C)
{
    const uint BK = 8;
    const uint TM = 8;
    const uint TN = 8;

    if (M >= 128 && N >= 128)
    {
        const uint BM = 128;
        const uint BN = 128;
        dim3 gridDim( CEIL_DIV(N, BN), CEIL_DIV(M, BM) );
        dim3 blockDim( (BM * BN) / (TM * TN) );   //  (128*128)/(8*8) = 256 threads
        sgemm2DBlocktiling<BM, BN, BK, TM, TN>
            <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
    }
    else
    {
        // fallback for small problems (still a power‑of‑two tile size)
        const uint BM = 64;
        const uint BN = 64;
        dim3 gridDim( CEIL_DIV(N, BN), CEIL_DIV(M, BM) );
        dim3 blockDim( (BM * BN) / (TM * TN) );   //  (64*64)/(8*8) = 64 threads
        sgemm2DBlocktiling<BM, BN, BK, TM, TN>
            <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
    }
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
    std::cout << "Running 2D-Block-Tiling Implementation" << std::endl;

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
        runSgemm2DBlocktiling(M, N, K, alpha, dA, dB, beta, dC);
        CUDA_CHECK(cudaDeviceSynchronize());

        // ----------------------------------------------------------
        // Timing with CUDA events
        // ----------------------------------------------------------
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        CUDA_CHECK(cudaEventRecord(start));
        for (int r = 0; r < repeat; ++r) {
            runSgemm2DBlocktiling(M, N, K, alpha, dA, dB, beta, dC);
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
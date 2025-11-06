//==========================================================================
//  vectorize.cu – GEMM with 2‑D block tiling and float4 vectorisation
//==========================================================================

#include <iostream>
#include <random>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdint>          // for uint32_t
#include <cassert>
#include <cstdio>
#include <cstdlib>

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

#define CUBLAS_CHECK(err)                                                 \
    do {                                                                  \
        cublasStatus_t s = (err);                                        \
        if (s != CUBLAS_STATUS_SUCCESS) {                                 \
            std::cerr << "cuBLAS error " << __FILE__ << ':' << __LINE__    \
                      << std::endl;                                      \
            std::exit(EXIT_FAILURE);                                      \
        }                                                                 \
    } while (0)

// ---------------------------------------------------------------------
// Integer ceiling division (kept for consistency with the other files)
// ---------------------------------------------------------------------
#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))

// ---------------------------------------------------------------------
// Kernel – 2‑D block tiling with vectorised loads/stores
// ---------------------------------------------------------------------
template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void __launch_bounds__((BM * BN) / (TM * TN), 1)
Vectorize(int M, int N, int K,
               float alpha, float *A,
               float *B, float beta, float *C)
{
    // -------------------------------------------------------------
    // 1) Identify the output tile this block will compute
    // -------------------------------------------------------------
    const uint cRow = blockIdx.y;                     // tile‑row index
    const uint cCol = blockIdx.x;                     // tile‑col index

    // BN/TN threads span a column of the tile
    const int threadCol = threadIdx.x % (BN / TN);
    const int threadRow = threadIdx.x / (BN / TN);

    // -------------------------------------------------------------
    // 2) Shared‑memory tiles for A and B
    // -------------------------------------------------------------
    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    // Move the global pointers to the first element of this tile
    A += cRow * BM * K;                     // start of A‑tile rows
    B += cCol * BN;                         // start of B‑tile columns
    C += cRow * BM * N + cCol * BN;         // start of C‑tile

    // -------------------------------------------------------------
    // 3) Indices used for loading the shared‑memory tiles
    //    (vectorised: 4 floats == one float4 per thread)
    // -------------------------------------------------------------
    const uint innerRowA = threadIdx.x / (BK / 4);
    const uint innerColA = threadIdx.x % (BK / 4);
    const uint innerRowB = threadIdx.x / (BN / 4);
    const uint innerColB = threadIdx.x % (BN / 4);

    // -------------------------------------------------------------
    // 4) Thread‑local accumulators (register‑blocked)
    // -------------------------------------------------------------
    float threadResults[TM * TN] = {0.0f};
    float regM[TM] = {0.0f};
    float regN[TN] = {0.0f};

    // -------------------------------------------------------------
    // 5) Main K‑loop – load tiles, compute, advance pointers
    // -------------------------------------------------------------
    for (uint bkIdx = 0; bkIdx < K; bkIdx += BK)
    {
        // ---- load a BM×BK tile of A (transpose while loading) ----
        float4 tmpA = reinterpret_cast<float4 *>(&A[innerRowA * K + innerColA * 4])[0];
        As[(innerColA * 4 + 0) * BM + innerRowA] = tmpA.x;
        As[(innerColA * 4 + 1) * BM + innerRowA] = tmpA.y;
        As[(innerColA * 4 + 2) * BM + innerRowA] = tmpA.z;
        As[(innerColA * 4 + 3) * BM + innerRowA] = tmpA.w;

        // ---- load a BK×BN tile of B (no transpose) ---------------
        reinterpret_cast<float4 *>(&Bs[innerRowB * BN + innerColB * 4])[0] =
            reinterpret_cast<float4 *>(&B[innerRowB * N + innerColB * 4])[0];

        __syncthreads();

        // ---- advance global pointers to the next K‑chunk ----------
        A += BK;          // move right by BK columns in A
        B += BK * N;      // move down by BK rows in B

        // ---- compute the partial products for this K‑chunk -------
        for (uint dotIdx = 0; dotIdx < BK; ++dotIdx)
        {
            // load TM elements of the current column of A into registers
            for (uint i = 0; i < TM; ++i)
                regM[i] = As[dotIdx * BM + threadRow * TM + i];

            // load TN elements of the current row of B into registers
            for (uint i = 0; i < TN; ++i)
                regN[i] = Bs[dotIdx * BN + threadCol * TN + i];

            // accumulate TM × TN products
            for (uint m = 0; m < TM; ++m)
                for (uint n = 0; n < TN; ++n)
                    threadResults[m * TN + n] += regM[m] * regN[n];
        }

        __syncthreads();   // ensure all threads finished before next tile
    }

    // -------------------------------------------------------------
    // 6) Write the TM × TN results back to global memory (vectorised)
    // -------------------------------------------------------------
    for (uint m = 0; m < TM; ++m)
    {
        for (uint n = 0; n < TN; n += 4)   // store 4 floats at a time
        {
            // load existing C values as a float4
            float4 tmpC = reinterpret_cast<float4 *>(
                &C[(threadRow * TM + m) * N + threadCol * TN + n])[0];

            // GEMM update
            tmpC.x = alpha * threadResults[m * TN + n]     + beta * tmpC.x;
            tmpC.y = alpha * threadResults[m * TN + n + 1] + beta * tmpC.y;
            tmpC.z = alpha * threadResults[m * TN + n + 2] + beta * tmpC.z;
            tmpC.w = alpha * threadResults[m * TN + n + 3] + beta * tmpC.w;

            // write back
            reinterpret_cast<float4 *>(
                &C[(threadRow * TM + m) * N + threadCol * TN + n])[0] = tmpC;
        }
    }
}

// ---------------------------------------------------------------------
// Launcher – mirrors the style of the other kernels
// ---------------------------------------------------------------------
void runVectorize(int M, int N, int K,
                       float alpha, float *A, float *B,
                       float beta, float *C)
{
    constexpr uint BK = 8;
    constexpr uint TM = 8;
    constexpr uint TN = 8;

    if (M >= 128 && N >= 128)
    {
        constexpr uint BM = 128;
        constexpr uint BN = 128;
        dim3 gridDim( CEIL_DIV(N, BN), CEIL_DIV(M, BM) );
        dim3 blockDim( (BM * BN) / (TM * TN) );   // (128*128)/(8*8) = 256 threads
        Vectorize<BM, BN, BK, TM, TN>
            <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
    }
    else
    {
        // fallback for small problems (still a power‑of‑two tile size)
        constexpr uint BM = 64;
        constexpr uint BN = 64;
        dim3 gridDim( CEIL_DIV(N, BN), CEIL_DIV(M, BM) );
        dim3 blockDim( (BM * BN) / (TM * TN) );   // (64*64)/(8*8) = 64 threads
        Vectorize<BM, BN, BK, TM, TN>
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
// Main driver – prints the exact format required by the assignment
// ---------------------------------------------------------------------
int main()
{
    std::cout << "Running Vectorized 2D‑Block‑Tiling Implementation" << std::endl;

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
    // Allocate device buffers (same maximum size) and copy once
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
        runVectorize(M, N, K, alpha, dA, dB, beta, dC);
        CUDA_CHECK(cudaDeviceSynchronize());

        // ----------------------------------------------------------
        // Timing with CUDA events
        // ----------------------------------------------------------
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        CUDA_CHECK(cudaEventRecord(start));
        for (int r = 0; r < repeat; ++r) {
            runVectorize(M, N, K, alpha, dA, dB, beta, dC);
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
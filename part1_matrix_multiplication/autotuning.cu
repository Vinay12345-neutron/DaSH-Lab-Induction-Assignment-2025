//==========================================================================
// autotuning.cu – GEMM with an auto‑tuned 2‑D block‑tiling kernel
//==========================================================================

#ifndef uint
#define uint unsigned int
#endif

#include <iostream>
#include <random>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cassert>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

// ---------------------------------------------------------------------
// Simple error‑checking macro
// ---------------------------------------------------------------------
#define CUDA_CHECK(err)                                                   \
    do {                                                                 \
        cudaError_t e = (err);                                           \
        if (e != cudaSuccess) {                                          \
            std::cerr << "CUDA error " << __FILE__ << ':' << __LINE__     \
                      << " : " << cudaGetErrorString(e) << std::endl;     \
            std::exit(EXIT_FAILURE);                                    \
        }                                                                \
    } while (0)

// ---------------------------------------------------------------------
// Kernel – auto‑tuned 2‑D block tiling
// ---------------------------------------------------------------------
constexpr int K9_NUM_THREADS = 256;

template <const int BM, const int BN, const int BK,
          const int TM, const int TN>
__global__ void __launch_bounds__(K9_NUM_THREADS, 1)
sgemmAutotuned(int M, int N, int K,
               float alpha, const float *A, const float *B,
               float beta,  float *C)
{
    const uint cRow = blockIdx.y;
    const uint cCol = blockIdx.x;

    // -----------------------------------------------------------------
    // Warptile dimensions (each warp = 16 threads)
    // -----------------------------------------------------------------
    constexpr int WM = TM * 16;
    constexpr int WN = TN * 16;
    constexpr int WMITER = CEIL_DIV(BM, WM);
    constexpr int WNITER = CEIL_DIV(BN, WN);

    // Thread position inside a warptile
    const int threadCol = threadIdx.x % (WN / TN);
    const int threadRow = threadIdx.x / (WN / TN);

    // Shared‑memory tiles
    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    // Point to the first tile of this block
    A += cRow * BM * K;
    B += cCol * BN;
    C += cRow * BM * N + cCol * BN;

    // -----------------------------------------------------------------
    // Indices for loading 4‑element vectors (float4) from global memory
    // -----------------------------------------------------------------
    const uint innerRowA = threadIdx.x / (BK / 4);
    const uint innerColA = threadIdx.x % (BK / 4);
    constexpr uint rowStrideA = (K9_NUM_THREADS * 4) / BK;

    const uint innerRowB = threadIdx.x / (BN / 4);
    const uint innerColB = threadIdx.x % (BN / 4);
    constexpr uint rowStrideB = K9_NUM_THREADS / (BN / 4);

    // -----------------------------------------------------------------
    // Per‑thread accumulator (register file)
    // -----------------------------------------------------------------
    float threadResults[WMITER * WNITER * TM * TN] = {0.0f};
    float regM[TM] = {0.0f};
    float regN[TN] = {0.0f};

    // -----------------------------------------------------------------
    // Main K‑loop – load tiles, compute, advance pointers
    // -----------------------------------------------------------------
    for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
        // ---- load A tile (transpose while storing) --------------------
        for (uint offset = 0; offset + rowStrideA <= BM; offset += rowStrideA) {
            float4 tmp = reinterpret_cast<const float4 *>(
                &A[(innerRowA + offset) * K + innerColA * 4])[0];

            As[(innerColA * 4 + 0) * BM + innerRowA + offset] = tmp.x;
            As[(innerColA * 4 + 1) * BM + innerRowA + offset] = tmp.y;
            As[(innerColA * 4 + 2) * BM + innerRowA + offset] = tmp.z;
            As[(innerColA * 4 + 3) * BM + innerRowA + offset] = tmp.w;
        }

        // ---- load B tile ------------------------------------------------
        for (uint offset = 0; offset + rowStrideB <= BK; offset += rowStrideB) {
            reinterpret_cast<float4 *>(
                &Bs[(innerRowB + offset) * BN + innerColB * 4])[0] =
                reinterpret_cast<const float4 *>(
                    &B[(innerRowB + offset) * N + innerColB * 4])[0];
        }

        __syncthreads();

        // ---- compute ----------------------------------------------------
        for (uint wmIdx = 0; wmIdx < WMITER; ++wmIdx) {
            for (uint wnIdx = 0; wnIdx < WNITER; ++wnIdx) {
                for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
                    // load a column of A into registers
                    for (uint i = 0; i < TM; ++i)
                        regM[i] = As[dotIdx * BM +
                                   (wmIdx * WM) + threadRow * TM + i];

                    // load a row of B into registers
                    for (uint i = 0; i < TN; ++i)
                        regN[i] = Bs[dotIdx * BN +
                                   (wnIdx * WN) + threadCol * TN + i];

                    // accumulate
                    for (uint m = 0; m < TM; ++m)
                        for (uint n = 0; n < TN; ++n)
                            threadResults[(wmIdx * TM + m) *
                                          (WNITER * TN) +
                                          wnIdx * TN + n] +=
                                regM[m] * regN[n];
                }
            }
        }

        __syncthreads();

        // advance to the next K‑tile
        A += BK;          // right by BK columns
        B += BK * N;      // down by BK rows
    }

    // -----------------------------------------------------------------
    // Write results back to global memory (vectorized stores)
    // -----------------------------------------------------------------
    for (uint wmIdx = 0; wmIdx < WMITER; ++wmIdx) {
        for (uint wnIdx = 0; wnIdx < WNITER; ++wnIdx) {
            float *C_interim = C + (wmIdx * WM * N) + (wnIdx * WN);

            for (uint m = 0; m < TM; ++m) {
                for (uint n = 0; n < TN; n += 4) {
                    // load 4‑element vector of C
                    float4 tmp = reinterpret_cast<float4 *>(
                        &C_interim[(threadRow * TM + m) * N +
                                   threadCol * TN + n])[0];

                    // update with GEMM result
                    const int idxBase = (wmIdx * TM + m) *
                                       (WNITER * TN) + wnIdx * TN + n;
                    tmp.x = alpha * threadResults[idxBase + 0] + beta * tmp.x;
                    tmp.y = alpha * threadResults[idxBase + 1] + beta * tmp.y;
                    tmp.z = alpha * threadResults[idxBase + 2] + beta * tmp.z;
                    tmp.w = alpha * threadResults[idxBase + 3] + beta * tmp.w;

                    // store back
                    reinterpret_cast<float4 *>(
                        &C_interim[(threadRow * TM + m) * N +
                                   threadCol * TN + n])[0] = tmp;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------
// Launcher – selects the tuned parameters for the target GPU
// ---------------------------------------------------------------------
void runSgemmAutotuned(int M, int N, int K,
                       float alpha, float *A, float *B,
                       float beta,  float *C)
{
    // Parameters tuned for an RTX A6000 (FP32)
    const uint K9_BK = 16;
    const uint K9_TM = 8;
    const uint K9_TN = 8;
    const uint K9_BM = 128;
    const uint K9_BN = 128;

    dim3 blockDim(K9_NUM_THREADS);
    dim3 gridDim(CEIL_DIV(N, K9_BN), CEIL_DIV(M, K9_BM));

    // sanity checks (identical to the original source)
    static_assert((K9_NUM_THREADS * 4) % K9_BK == 0,
                  "NUM_THREADS*4 must be multiple of K9_BK");
    static_assert((K9_NUM_THREADS * 4) % K9_BN == 0,
                  "NUM_THREADS*4 must be multiple of K9_BN");
    static_assert(K9_BN % (16 * K9_TN) == 0,
                  "K9_BN must be a multiple of 16*K9_TN");
    static_assert(K9_BM % (16 * K9_TM) == 0,
                  "K9_BM must be a multiple of 16*K9_TM");
    static_assert((K9_BM * K9_BK) % (4 * K9_NUM_THREADS) == 0,
                  "K9_BM*K9_BK must be a multiple of 4*NUM_THREADS");
    static_assert((K9_BN * K9_BK) % (4 * K9_NUM_THREADS) == 0,
                  "K9_BN*K9_BK must be a multiple of 4*NUM_THREADS");

    sgemmAutotuned<K9_BM, K9_BN, K9_BK, K9_TM, K9_TN>
        <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}

// ---------------------------------------------------------------------
// Helper utilities (same as the other samples)
// ---------------------------------------------------------------------
static void random_fill(float *p, size_t n)
{
    std::mt19937 rng(0);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (size_t i = 0; i < n; ++i) p[i] = dist(rng);
}

static double gflops(int M, int N, int K, double sec)
{
    return 2.0 * M * N * K / (sec * 1e9);
}

// ---------------------------------------------------------------------
// Main driver – identical benchmarking harness to the other examples
// ---------------------------------------------------------------------
int main()
{
    std::cout << "Running Auto‑tuned GEMM (2D block tiling)" << std::endl;

    const int maxSize = 4096;
    const int sizes[] = {128, 256, 512, 1024, 2048, 4096};
    const int nSizes = sizeof(sizes) / sizeof(sizes[0]);

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

    const float alpha = 0.5f;
    const float beta  = 3.0f;
    const int   repeat = 50;

    for (int i = 0; i < nSizes; ++i) {
        int M = sizes[i];
        int N = sizes[i];
        int K = sizes[i];

        // warm‑up
        runSgemmAutotuned(M, N, K, alpha, dA, dB, beta, dC);
        CUDA_CHECK(cudaDeviceSynchronize());

        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        CUDA_CHECK(cudaEventRecord(start));
        for (int r = 0; r < repeat; ++r) {
            runSgemmAutotuned(M, N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= repeat;

        double sec = ms * 1e-3;
        double perf = gflops(M, N, K, sec);

        printf("dimensions(m=n=k) %d, alpha: %.1f, beta: %.1f\n",
               M, alpha, beta);
        printf("Average elapsed time: (%.6f) s, performance: (%8.1f) GFLOPS. size: (%d).\n",
               sec, perf, M);

        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
    }

    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));
    free(hA);
    free(hB);
    free(hC);
    return 0;
}
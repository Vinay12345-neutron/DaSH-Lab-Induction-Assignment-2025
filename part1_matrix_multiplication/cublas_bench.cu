#ifndef uint
#define uint unsigned int
#endif

#include <iostream>
#include <random>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdint>          // for uint32_t
#include <cassert>

// ---------------------------------------------------------------------
// Simple error‑checking macros (CUDA and cuBLAS)
// ---------------------------------------------------------------------
#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err = (call);                                          \
        if (err != cudaSuccess) {                                          \
            std::cerr << "CUDA error " << __FILE__ << ':' << __LINE__       \
                      << " : " << cudaGetErrorString(err) << std::endl;     \
            std::exit(EXIT_FAILURE);                                      \
        }                                                                  \
    } while (0)

#define CUBLAS_CHECK(call)                                                 \
    do {                                                                   \
        cublasStatus_t stat = (call);                                      \
        if (stat != CUBLAS_STATUS_SUCCESS) {                               \
            std::cerr << "cuBLAS error " << __FILE__ << ':' << __LINE__     \
                      << std::endl;                                        \
            std::exit(EXIT_FAILURE);                                      \
        }                                                                  \
    } while (0)

// ---------------------------------------------------------------------
// Integer ceiling division (kept for consistency with other files)
// ---------------------------------------------------------------------
#define CEIL_DIV(x, y)   ( ((x) + (y) - 1) / (y) )

// ---------------------------------------------------------------------
// Fill a buffer with deterministic pseudo‑random numbers [0,1)
// Column‑major layout (element (r,c) is at p[c*rows + r])
// ---------------------------------------------------------------------
static void random_fill_colmajor(float *p, size_t rows, size_t cols)
{
    std::mt19937 rng(0);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (size_t c = 0; c < cols; ++c)
        for (size_t r = 0; r < rows; ++r)
            p[c * rows + r] = dist(rng);
}

// ---------------------------------------------------------------------
// Compute GFLOPS for a single SGEMM of size M×N×K
// ---------------------------------------------------------------------
static double gflops(int M, int N, int K, double seconds)
{
    double flops = 2.0 * static_cast<double>(M) *
                         static_cast<double>(N) *
                         static_cast<double>(K);
    return (flops * 1e-9) / seconds;   // GFLOP/s
}

// ---------------------------------------------------------------------
// Main driver 
// ---------------------------------------------------------------------
int main()
{
    std::cout << "Running cuBLAS SGEMM Implementation" << std::endl;

    const float alpha = 0.5f;
    const float beta  = 3.0f;
    const int   repeat = 50;               // timed launches per size
    const int   warm_up = 5;               // warm‑up launches

    // -----------------------------------------------------------------
    // Square matrix sizes to benchmark
    // -----------------------------------------------------------------
    const int sizes[] = {128, 256, 512, 1024, 2048, 4096};
    const int nSizes   = sizeof(sizes) / sizeof(sizes[0]);

    // -----------------------------------------------------------------
    // Allocate host buffers for the *maximum* matrix size (4096×4096)
    // -----------------------------------------------------------------
    const int maxSize = 4096;
    const size_t maxElems = static_cast<size_t>(maxSize) * maxSize;

    float *hA = (float*)malloc(maxElems * sizeof(float));
    float *hB = (float*)malloc(maxElems * sizeof(float));
    float *hC = (float*)malloc(maxElems * sizeof(float));
    if (!hA || !hB || !hC) {
        std::cerr << "Host allocation failed\n";
        return EXIT_FAILURE;
    }

    // Initialise with deterministic data (column‑major)
    random_fill_colmajor(hA, maxSize, maxSize);
    random_fill_colmajor(hB, maxSize, maxSize);
    // hC will be overwritten by cuBLAS, no need to initialise

    // -----------------------------------------------------------------
    // Allocate device buffers (same maximum size)
    // -----------------------------------------------------------------
    float *dA, *dB, *dC;
    CUDA_CHECK(cudaMalloc(&dA, maxElems * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dB, maxElems * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dC, maxElems * sizeof(float)));

    // Transfer the *largest* matrices once; smaller sizes will just use a
    // prefix of these buffers.
    CUDA_CHECK(cudaMemcpy(dA, hA, maxElems * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB, maxElems * sizeof(float),
                          cudaMemcpyHostToDevice));

    // -----------------------------------------------------------------
    // Create cuBLAS handle (single handle reused for all sizes)
    // -----------------------------------------------------------------
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    // -----------------------------------------------------------------
    // Loop over the requested matrix sizes
    // -----------------------------------------------------------------
    for (int i = 0; i < nSizes; ++i) {
        const int M = sizes[i];
        const int N = sizes[i];
        const int K = sizes[i];

        // -------------------------------------------------------------
        // Warm‑up launches (remove first‑run overhead)
        // -------------------------------------------------------------
        for (int w = 0; w < warm_up; ++w) {
            CUBLAS_CHECK(cublasSgemm(handle,
                                     CUBLAS_OP_N, CUBLAS_OP_N,
                                     N, M, K,               // note: cuBLAS is column‑major
                                     &alpha,
                                     dB, N,                 // B is N×K
                                     dA, K,                 // A is K×M
                                     &beta,
                                     dC, N));                // C is N×M
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        // -------------------------------------------------------------
        // Timing with CUDA events
        // -------------------------------------------------------------
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        CUDA_CHECK(cudaEventRecord(start));
        for (int r = 0; r < repeat; ++r) {
            CUBLAS_CHECK(cublasSgemm(handle,
                                     CUBLAS_OP_N, CUBLAS_OP_N,
                                     N, M, K,
                                     &alpha,
                                     dB, N,
                                     dA, K,
                                     &beta,
                                     dC, N));
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float msTotal = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&msTotal, start, stop));
        double secPer = static_cast<double>(msTotal) / 1000.0 / repeat; // seconds

        double perf = gflops(M, N, K, secPer);

        // ----- EXACT output format -----
        printf("dimensions(m=n=k) %d, alpha: %.1f, beta: %.1f\n",
               M, alpha, beta);
        printf("Average elapsed time: (%.6f) s, performance: (%8.1f) GFLOPS. size: (%d).\n",
               secPer, perf, M);
        // ------------------------------

        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
    }

    // -----------------------------------------------------------------
    // Clean‑up
    // -----------------------------------------------------------------
    CUBLAS_CHECK(cublasDestroy(handle));
    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));
    free(hA);
    free(hB);
    free(hC);

    return 0;
}
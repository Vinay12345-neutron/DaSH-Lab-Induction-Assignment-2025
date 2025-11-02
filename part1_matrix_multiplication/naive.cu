#include <iostream>
#include <random>
#include <cuda_runtime.h>

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
// Naïve kernel (row‑major, one thread per output element)
// ---------------------------------------------------------------------
__global__ void naive(int M, int N, int K,
                            float alpha,
                            const float* A,
                            const float* B,
                            float beta,
                            float* C)
{
    const uint x = blockIdx.x * blockDim.x + threadIdx.x; // row
    const uint y = blockIdx.y * blockDim.y + threadIdx.y; // col

    if (x < M && y < N) {
        float acc = 0.0f;
        for (int i = 0; i < K; ++i){
            acc += A[x * K + i] * B[i * N + y];
        }
            
        C[x * N + y] = alpha * acc + beta * C[x * N + y];
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

// --------------------------------------------------------------------
// Main driver 
// ---------------------------------------------------------------------
int main()
{
    std::cout << "Running Naive Implementation" << std::endl;

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

    const dim3 block(16, 16);
    const float alpha = 0.5f;
    const float beta  = 3.0f;
    const int repeat = 50;               // timed launches per size

    // --------------------------------------------------------------
    // Loop over the six matrix sizes
    // --------------------------------------------------------------
    for (int i = 0; i < nSizes; ++i) {
        int M = sizes[i];
        int N = sizes[i];
        int K = sizes[i];

        dim3 grid( (M + block.x - 1) / block.x,
                   (N + block.y - 1) / block.y );

        // Warm‑up (removes first‑run overhead)
        naive<<<grid, block>>>(M, N, K, alpha, dA, dB, beta, dC);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Timing with CUDA events
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        CUDA_CHECK(cudaEventRecord(start));
        for (int r = 0; r < repeat; ++r)
            naive<<<grid, block>>>(M, N, K, alpha, dA, dB, beta, dC);
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
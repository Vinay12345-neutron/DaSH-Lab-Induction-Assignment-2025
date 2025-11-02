// ---------------------------------------------------------------
// Minimal cuBLAS SGEMM benchmark (FP32)
// ---------------------------------------------------------------
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CUDA_CHECK(call) do {                                 \
    cudaError_t err = call;                                   \
    if (err != cudaSuccess) {                                 \
        fprintf(stderr, "CUDA error %s:%d: %s\n",              \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE);                                   \
    } } while (0)

#define CUBLAS_CHECK(call) do {                               \
    cublasStatus_t stat = call;                               \
    if (stat != CUBLAS_STATUS_SUCCESS) {                      \
        fprintf(stderr, "cuBLAS error %s:%d\n",               \
                __FILE__, __LINE__);                          \
        exit(EXIT_FAILURE);                                   \
    } } while (0)

int main()
{
    const int M = 4096, N = 4096, K = 4096;
    const float alpha = 1.0f, beta = 0.0f;
    const size_t bytesA = (size_t)M * K * sizeof(float);
    const size_t bytesB = (size_t)K * N * sizeof(float);
    const size_t bytesC = (size_t)M * N * sizeof(float);
    const int nIter = 30;          // enough for a stable average

    // ---- host allocation & init ---------------------------------
    float *hA = (float*)malloc(bytesA);
    float *hB = (float*)malloc(bytesB);
    float *hC = (float*)malloc(bytesC);
    for (size_t i = 0; i < bytesA/sizeof(float); ++i) hA[i] = 1.0f;
    for (size_t i = 0; i < bytesB/sizeof(float); ++i) hB[i] = 1.0f;

    // ---- device allocation --------------------------------------
    float *dA, *dB, *dC;
    CUDA_CHECK(cudaMalloc(&dA, bytesA));
    CUDA_CHECK(cudaMalloc(&dB, bytesB));
    CUDA_CHECK(cudaMalloc(&dC, bytesC));

    CUDA_CHECK(cudaMemcpy(dA, hA, bytesA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB, bytesB, cudaMemcpyHostToDevice));

    // ---- cuBLAS handle -----------------------------------------
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    // ---- warm‑up ------------------------------------------------
    CUBLAS_CHECK(cublasSgemm(handle,
                             CUBLAS_OP_N, CUBLAS_OP_N,
                             N, M, K,
                             &alpha,
                             dB, N,          // note column‑major ordering
                             dA, K,
                             &beta,
                             dC, N));
    CUDA_CHECK(cudaDeviceSynchronize());

    // ---- timing ------------------------------------------------
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < nIter; ++i) {
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
    float msPer = msTotal / nIter;

    double flops = 2.0 * (double)M * (double)N * (double)K;
    double gflops = (flops * 1e-9) / (msPer / 1000.0);

    printf("cuBLAS SGEMM  %dx%dx%d  :  %8.3f ms  →  %8.2f GFLOP/s\n",
           M, N, K, msPer, gflops);

    // ---- cleanup ------------------------------------------------
    CUBLAS_CHECK(cublasDestroy(handle));
    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));
    free(hA); free(hB); free(hC);
    return 0;
}
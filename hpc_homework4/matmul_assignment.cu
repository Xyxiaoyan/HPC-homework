// CUDA matrix multiplication assignment. The same file also builds as CPU-only C++:
//   g++ -O3 -std=c++17 -x c++ matmul_assignment.cu -o matmul_cpu
//   nvcc -O3 -std=c++14 matmul_assignment.cu -o matmul_cuda
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#ifdef __CUDACC__
#include <cuda_runtime.h>
#endif

using Clock = std::chrono::steady_clock;

static double median(std::vector<double> x) {
    std::sort(x.begin(), x.end());
    return x[x.size() / 2];
}

// A and B are N x N, row-major. i-k-j order makes the CPU baseline cache friendly.
static void matmul_cpu(const float* A, const float* B, float* C, int n) {
    std::fill(C, C + static_cast<size_t>(n) * n, 0.0f);
    for (int i = 0; i < n; ++i)
        for (int k = 0; k < n; ++k) {
            const float a = A[static_cast<size_t>(i) * n + k];
            for (int j = 0; j < n; ++j)
                C[static_cast<size_t>(i) * n + j] += a * B[static_cast<size_t>(k) * n + j];
        }
}

#ifdef __CUDACC__
#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        std::cerr << #call << ": " << cudaGetErrorString(err) << '\n'; \
        std::exit(EXIT_FAILURE); \
    } \
} while (0)

// Based on the course's naive example: one thread computes one output element.
__global__ void matmul_naive(const float* A, const float* B, float* C, int n) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n && col < n) {
        float sum = 0.0f;
        for (int k = 0; k < n; ++k)
            sum += A[static_cast<size_t>(row) * n + k] * B[static_cast<size_t>(k) * n + col];
        C[static_cast<size_t>(row) * n + col] = sum;
    }
}

static void matmul_gpu(const std::vector<float>& A, const std::vector<float>& B,
                       std::vector<float>& C, int n, double& kernel_ms,
                       double& total_ms) {
    const size_t bytes = A.size() * sizeof(float);
    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_A), bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_B), bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_C), bytes));

    cudaEvent_t start, kernel_start, kernel_stop, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&kernel_start));
    CUDA_CHECK(cudaEventCreate(&kernel_stop));
    CUDA_CHECK(cudaEventCreate(&stop));
    const dim3 block(16, 16);
    const dim3 grid((n + 15) / 16, (n + 15) / 16);
    std::vector<double> kernels, totals;

    // One warm-up run excludes one-time context and JIT initialization.
    CUDA_CHECK(cudaMemcpy(d_A, A.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B.data(), bytes, cudaMemcpyHostToDevice));
    matmul_naive<<<grid, block>>>(d_A, d_B, d_C, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    for (int rep = 0; rep < 3; ++rep) {
        CUDA_CHECK(cudaEventRecord(start));
        CUDA_CHECK(cudaMemcpy(d_A, A.data(), bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B, B.data(), bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaEventRecord(kernel_start));
        matmul_naive<<<grid, block>>>(d_A, d_B, d_C, n);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(kernel_stop));
        CUDA_CHECK(cudaMemcpy(C.data(), d_C, bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float km = 0, tm = 0;
        CUDA_CHECK(cudaEventElapsedTime(&km, kernel_start, kernel_stop));
        CUDA_CHECK(cudaEventElapsedTime(&tm, start, stop));
        kernels.push_back(km);
        totals.push_back(tm);
    }
    kernel_ms = median(kernels);
    total_ms = median(totals);
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(kernel_start));
    CUDA_CHECK(cudaEventDestroy(kernel_stop));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
}
#endif

int main(int argc, char** argv) {
    std::vector<int> sizes;
    if (argc == 1) sizes = {128, 256, 512, 1024};
    for (int i = 1; i < argc; ++i) {
        try {
            const std::string arg(argv[i]);
            size_t used = 0;
            const long value = std::stol(arg, &used);
            if (used != arg.size() || value < 1 || value > 4096)
                throw std::out_of_range("size");
            sizes.push_back(static_cast<int>(value));
        } catch (...) {
            std::cerr << "Matrix sizes must be integers from 1 to 4096.\n";
            return EXIT_FAILURE;
        }
    }

#ifdef __CUDACC__
    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count == 0) {
        std::cerr << "No CUDA device found.\n";
        return EXIT_FAILURE;
    }
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::cerr << "GPU: " << prop.name << '\n';
#else
    std::cerr << "CPU-only build: GPU measurements are unavailable.\n";
#endif

    std::cout << "N,cpu_ms,gpu_kernel_ms,gpu_total_ms,kernel_speedup,total_speedup\n";
    for (int n : sizes) {
        const size_t elements = static_cast<size_t>(n) * n;
        std::vector<float> A(elements), B(elements), cpu(elements);
        for (size_t i = 0; i < elements; ++i) {
            A[i] = static_cast<float>(i % 13 + 1) / 13.0f;
            B[i] = static_cast<float>(i % 17 + 1) / 17.0f;
        }
        std::vector<double> cpu_times;
        for (int rep = 0; rep < 3; ++rep) {
            const auto start = Clock::now();
            matmul_cpu(A.data(), B.data(), cpu.data(), n);
            const auto stop = Clock::now();
            cpu_times.push_back(std::chrono::duration<double, std::milli>(stop - start).count());
        }
        const double cpu_ms = median(cpu_times);
        std::cout << n << ',' << cpu_ms;
#ifdef __CUDACC__
        std::vector<float> gpu(elements);
        double kernel_ms = 0, total_ms = 0;
        matmul_gpu(A, B, gpu, n, kernel_ms, total_ms);
        std::cout << ',' << kernel_ms << ',' << total_ms << ','
                  << cpu_ms / kernel_ms << ',' << cpu_ms / total_ms << '\n';
#else
        std::cout << ",N/A,N/A,N/A,N/A\n";
#endif
    }
}

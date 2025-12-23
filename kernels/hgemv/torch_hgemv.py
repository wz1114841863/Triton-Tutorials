import torch
from torch.utils.cpp_extension import load_inline

cuda_source = """
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <float.h>
#include <cuda_fp16.h>

#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])
#define HALF2(value) (reinterpret_cast<half2 *>(&(value))[0])

// Half 版本的 Warp Reduce
__device__ __forceinline__ half warp_reduce_sum_f16(half val) {
    // 实现规约
    for (int offset = 16; offset > 0; offset /= 2) {
        val = __hadd(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

// Float 版本的 Warp Reduce
__device__ __forceinline__ float warp_reduce_sum_f32(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// FP16 HGEMV
// 假设 K 是 32 的倍数
// BlockDim = (32, 4)
__global__ void hgemv_k32_f16_kernel(half *a, half *x, half *y, int M, int K) {
    // 每个 block 处理 4 行
    int row = blockIdx.x * 4 + threadIdx.y;
    if (row >= M) return;

    half sum = __float2half(0.0f);
    for (int k = threadIdx.x; k < K; k += 32) {
        half a_val = a[row * K + k];
        half x_val = x[k];
        sum = __hadd(sum, __hmul(a_val, x_val));
    }

    // Warp Reduce
    sum = warp_reduce_sum_f16(sum);

    // Thread 0 写回结果
    if (threadIdx.x == 0) {
        y[row] = sum;
    }
}

// 精度更高的 FP16 HGEMV 实现
// 假设 K 是 32 的倍数
// BlockDim = (32, 4)
__global__ void hgemv_k32_f16_kernel_v2(half *a, half *x, half *y, int M, int K) {
    int row = blockIdx.x * 4 + threadIdx.y;
    if (row >= M) return;

    // 使用 float 做累加器
    float sum_f32 = 0.0f;

    for (int k = threadIdx.x; k < K; k += 32) {
        // 先转成 float 再计算乘加 (FMA)
        float a_val = __half2float(a[row * K + k]);
        float x_val = __half2float(x[k]);
        sum_f32 += a_val * x_val;
    }

    // Warp Reduce 也要用 float 版本
    sum_f32 = warp_reduce_sum_f32(sum_f32);

    if (threadIdx.x == 0) {
        // 最后写入时才转回 half
        y[row] = __float2half(sum_f32);
    }
}

// FP16 half2 SGEMV 实现
__global__ void hgemv_k128_f16x4_kernel(half *a, half *x, half *y, int M, int K) {
    int lane = threadIdx.x;
    int m = blockIdx.x * blockDim.y + threadIdx.y; // 行号

    if (m < M) {
        // 1. 定义两个 half2 累加器 (相当于 4 路同时累加)
        // half2 sum2_a = __float2half2_rn(0.0f); // 存第 1/2 个数的和
        // half2 sum2_b = __float2half2_rn(0.0f); // 存第 3/4 个数的和

        // 使用 float 作为累加器
        float sum_f32 = 0.0f;

        // 步长:每个 Warp 处理 128 个元素 (32 * 4)
        for (int k_base = lane * 4; k_base < K; k_base += 128) {
            // 向量化读取 x (利用 HALF2 宏)
            // 读取 x[k_base] 和 x[k_base+1] 组成一个 half2
            half2 x_0 = HALF2(x[k_base]);
            // 读取 x[k_base+2] 和 x[k_base+3] 组成一个 half2
            half2 x_1 = HALF2(x[k_base + 2]);

            // 向量化读取 A (利用 HALF2 宏)
            // A 的索引是 m * K + k_base
            half2 a_0 = HALF2(a[m * K + k_base]);
            half2 a_1 = HALF2(a[m * K + k_base + 2]);

            // SIMD 乘加 (FMA)
            // 有精度差异
            // sum2_a += x_0 * a_0
            // sum2_a = __hfma2(x_0, a_0, sum2_a);
            // sum2_b = __hfma2(x_1, a_1, sum2_b);

            // 转换为 float2 进行计算(关键修改)
            float2 x_f0 = __half22float2(x_0);
            float2 x_f1 = __half22float2(x_1);
            float2 a_f0 = __half22float2(a_0);
            float2 a_f1 = __half22float2(a_1);

            // 使用 float 进行累加
            sum_f32 += a_f0.x * x_f0.x + a_f0.y * x_f0.y;
            sum_f32 += a_f1.x * x_f1.x + a_f1.y * x_f1.y;
        }

        // 2. 将 half2 结果汇总为 float (为了精度和 Reduce)
        // float sum_f32 = 0.0f;

        // 把 sum2_a.x, sum2_a.y, sum2_b.x, sum2_b.y 累加到 sum_f32
        // sum_f32 += __low2float(sum2_a) + __high2float(sum2_a);
        //sum_f32 += __low2float(sum2_b) + __high2float(sum2_b);

        // 3. Warp Reduce (FP32)
        sum_f32 = warp_reduce_sum_f32(sum_f32);

        // 4. 写回
        if (lane == 0)
            y[m] = __float2half(sum_f32);
    }
}

torch::Tensor hgemv_k128_f16x4(torch::Tensor a, torch::Tensor x, torch::Tensor y) {
    const int M = a.size(0);
    const int K = a.size(1);

    // 默认K是4的倍数
    dim3 block(32, 4);
    dim3 grid((M + 4 - 1) / 4);

    hgemv_k128_f16x4_kernel<<<grid, block>>>(
        reinterpret_cast<half *>(a.data_ptr()),
        reinterpret_cast<half *>(x.data_ptr()),
        reinterpret_cast<half *>(y.data_ptr()),
        M, K
    );

    return y;
}
"""

cpp_source = (
    "torch::Tensor hgemv_k128_f16x4(torch::Tensor a, torch::Tensor x, torch::Tensor y);"
)

module = load_inline(
    name="hgemv_ops",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["hgemv_k128_f16x4"],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
)

# --- 测试 ---
M, N = 1024, 256
a = torch.randn(M, N, device="cuda")
x = torch.randn(N, device="cuda")
y = torch.empty(M, device="cuda")
# 转成 FP16
a = a.half()
x = x.half()
y = y.half()

# 运行自定义算子
y_my = module.hgemv_k128_f16x4(a, x, y)
# 运行 PyTorch 原生算子
y_torch = torch.mv(a, x)

max_diff = (y_my - y_torch).abs().max().item()
print(f"Max diff: {max_diff}")

# 2. 适当放宽标准
if torch.allclose(y_my, y_torch, atol=1e-5):
    print("✅ SGEMV Match!")
else:
    print("❌ Mismatch!")

import torch
from torch.utils.cpp_extension import load_inline

cuda_source = """
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <float.h>

#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])

__global__ void sgemv_naive_kernel(float* A, float* x, float* y, int M, int K) {
    // 计算当前线程负责的行号 row
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M)
        return;
    float result = 0.0f;
    unsigned int start = row * K;

    // 遍历这一行的 K 个元素,与 x 进行点积
    for (int col = 0; col < K; ++col) {
        result += A[start + col] * x[col];
    }
    // 将结果写入 y[row]
    y[row] = result;
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// 假设 K 是 32 的倍数,简化边界处理
// Block Dim: (32, 4).这意味着一个 Block 有 32 * 4 = 128 个线程.
// x 维度是 32:对应 Warp 里的 Lane ID,用来负责列方向K的并行.
// y 维度是 4:对应 Block 里的 Warp ID,用来负责行方向M的并行(即一个 Block 处理 4 行).
// Grid Dim: M / 4 (向上取整)
__global__ void sgemv_k32_kernel(float* A, float* x, float* y, int M, int K) {
    // 1. 算出 lane_id (0~31)
    int lane = threadIdx.x;

    // 2. 算出当前 Warp 负责的行号 row
    // 利用 blockIdx.x 和 threadIdx.y
    int row = blockIdx.x * blockDim.y + threadIdx.y;

    if (row < M) {
        float sum = 0.0f;

        // 3. 循环处理 K 维度 (Coalesced Access)
        // 每个线程间隔 32 读取
        for (int k = lane; k < K; k += 32) {
            sum += A[row * K + k] * x[k];
        }

        // 4. Warp 内归约
        sum = warp_reduce_sum(sum);

        // 5. 写回结果
        if (lane == 0) {
            y[row] = sum;
        }
    }
}



__global__ void sgemv_k128_f32x4_kernel(float *A, float *x, float *y, int M, int K) {
    // 1. 基础索引
    int lane = threadIdx.x;
    int row = blockIdx.x * blockDim.y + threadIdx.y;

    if (row < M) {
        float sum = 0.0f;

        // 2. 循环处理 K 维度
        // 现在的步长是 WARP_SIZE * 4 = 128
        // 我们用 k_base 表示 float4 的起始索引
        for (int k_base = lane * 4; k_base < K; k_base += 128) {

            // 向量化读取 x 的 4 个元素
            // x 的地址是 x + k_base
            float4 vec_x = FLOAT4(x[k_base]);

            // 向量化读取 A 的 4 个元素
            // A 当前行的地址是 A + row * K + k_base
            float4 vec_A = FLOAT4(A[row * K + k_base]);

            // 手动计算 float4 的点积并累加到 sum
            sum += vec_A.x * vec_x.x + vec_A.y * vec_x.y + vec_A.z * vec_x.z + vec_A.w * vec_x.w;
        }

        // 3. Warp 内归约
        sum = warp_reduce_sum(sum);

        // 4. 写回
        if (lane == 0) y[row] = sum;
    }
}

torch::Tensor sgemv_k128_f32x4(torch::Tensor a, torch::Tensor x, torch::Tensor y) {
    const int M = a.size(0);
    const int K = a.size(1);

    // 默认K是4的倍数
    dim3 block(32, 4);
    dim3 grid((M + 4 - 1) / 4);

    sgemv_k128_f32x4_kernel<<<grid, block>>>(
        reinterpret_cast<float *>(a.data_ptr()),
        reinterpret_cast<float *>(x.data_ptr()),
        reinterpret_cast<float *>(y.data_ptr()),
        M, K
    );

    return y;
}
"""

cpp_source = (
    "torch::Tensor sgemv_k128_f32x4(torch::Tensor a, torch::Tensor x, torch::Tensor y);"
)

module = load_inline(
    name="sgemv_ops",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["sgemv_k128_f32x4"],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
)

# --- 测试 ---
M, N = 1024, 256
a = torch.randn(M, N, device="cuda")
x = torch.randn(N, device="cuda")
y = torch.empty(M, device="cuda")

# 运行自定义算子
y_my = module.sgemv_k128_f32x4(a, x, y)
# 运行 PyTorch 原生算子
y_torch = torch.mv(a, x)

max_diff = (y_my - y_torch).abs().max().item()
print(f"Max diff: {max_diff}")

# 2. 适当放宽标准 (FP32 手写 Reduce 通常给 1e-5)
if torch.allclose(y_my, y_torch, atol=1e-5):
    print("✅ SGEMV Match!")
else:
    print("❌ Mismatch!")

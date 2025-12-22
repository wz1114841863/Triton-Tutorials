import torch
from torch.utils.cpp_extension import load_inline

cuda_source = """
#include <cuda_runtime.h>
#include <float.h>

#define WARP_SIZE 32

// Warp Reduce Max, 求出每个Warp内的最大值
__device__ __forceinline__ float warp_reduce_max(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, offset));
    }
    return val;
}

// Warp Reduce Sum, 求出每个Warp内的和
__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Block Reduce Max,
// 利用 Shared Memory 进行 Block 级别的 Max Reduce
__device__ __forceinline__ float block_reduce_max(float val) {
    static __shared__ float shared[32]; // 假设 max block size = 1024 (32 warps)
    int lane = threadIdx.x % WARP_SIZE;
    int wid = threadIdx.x / WARP_SIZE;

    // 1. Warp 内 Reduce
    val = warp_reduce_max(val);

    // 2. 只有 Warp 的第一个线程把结果写入 Shared Memory
    if (lane == 0) shared[wid] = val;
    __syncthreads(); // 等待所有 Warp 写完

    // 3. 由第一个 Warp 再次读取并 Reduce
    // (注意:这里假设 BlockSize <= 1024,即 Warp 数量 <= 32)
    val = (threadIdx.x < blockDim.x / WARP_SIZE) ? shared[lane] : -FLT_MAX;

    if (wid == 0)
        val = warp_reduce_max(val);

    // 广播给 Block 内所有线程
    // 简单做法:Warp 0 写回 Shared[0],大家读
    if (threadIdx.x == 0)
        shared[0] = val;
    __syncthreads();

    return shared[0];
}

// Block Reduce Sum
// 利用 Shared Memory 进行 Block 级别的 Sum Reduce
__device__ __forceinline__ float block_reduce_sum(float val) {
    static __shared__ float shared[32];
    int lane = threadIdx.x % WARP_SIZE;
    int wid = threadIdx.x / WARP_SIZE;

    val = warp_reduce_sum(val);
    if (lane == 0) shared[wid] = val;
    __syncthreads();

    // 假设 BlockSize 足够大, Warp 0 负责最后汇总
    val = (threadIdx.x < blockDim.x / WARP_SIZE) ? shared[lane] : 0.0f;
    if (wid == 0) val = warp_reduce_sum(val);

    if (threadIdx.x == 0) shared[0] = val;
    __syncthreads();

    return shared[0];
}

// Naive Softmax Kernel
__global__ void softmax_naive_kernel(float* input, float* output, int N) {
    // 假设 GridDim.x = M (行数), BlockDim.x = N (列数, <= 1024)
    // 每个 Block 处理一行

    int row_idx = blockIdx.x;
    int tid = threadIdx.x;

    // 1. 计算全局偏移
    // input_row 指向当前行起始位置
    float* input_row = input + row_idx * N;
    float* output_row = output + row_idx * N;

    // 2. 加载数据
    // 边界检查:如果 N < 1024,需要 if (tid < N)
    // 每个线程加载一个数
    float val = (tid < N) ? input_row[tid] : -FLT_MAX;

    // 调用 block_reduce_max 找到当前行的最大值
    float max_val = block_reduce_max(val);

    // 3. 计算 Exp & Sum
    // 计算 e^(x - max) 并累加
    val = (tid < N) ? expf(val - max_val) : 0.0f;
    float sum_val = block_reduce_sum(val);

    // 4. 归一化并写回
    if (tid < N) {
        output_row[tid] = val / sum_val;
    }
}

// Host Wrapper
torch::Tensor launch_softmax(torch::Tensor input) {
    int M = input.size(0);
    int N = input.size(1);
    auto output = torch::empty_like(input);

    // 限制:暂时只支持 N <= 1024
    int block_size = N;
    // 实际上应该向上取整到 32 的倍数,这里简化处理
    if (block_size > 1024) throw std::runtime_error("N must be <= 1024 for naive kernel");

    dim3 grid(M);
    dim3 block(block_size);
    // 动态 Shared Memory 大小?这里我们在 helper 里用了 static,所以不需要动态分配
    // 但实际生产中通常会动态分配

    softmax_naive_kernel<<<grid, block>>>(
        (float*)input.data_ptr(),
        (float*)output.data_ptr(),
        N
    );
    return output;
}
"""

cpp_source = "torch::Tensor launch_softmax(torch::Tensor input);"

module = load_inline(
    name="softmax_ops",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["launch_softmax"],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
)

# --- 测试 ---
M, N = 10, 256
x = torch.randn(M, N, device="cuda")

# 运行自定义算子
y_my = module.launch_softmax(x)
# 运行 PyTorch 原生算子
y_torch = torch.softmax(x, dim=1)

if torch.allclose(y_my, y_torch):
    print("✅ Softmax Match!")
else:
    print("❌ Mismatch!")
    print("My:", y_my[0, :5])
    print("PyTorch:", y_torch[0, :5])

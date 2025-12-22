import torch
from torch.utils.cpp_extension import load_inline

cuda_source = """
#include <cuda_runtime.h>
#include <float.h>

#define WARP_SIZE 32

__global__ void softmax_online_kernel(float* input, float* output, int N) {
    int row_idx = blockIdx.x;
    int tid = threadIdx.x;

    // 全局内存指针
    float* input_row = input + row_idx * N;
    float* output_row = output + row_idx * N;

    // 线程私有寄存器 (Register)
    // 负责存储当前线程处理的局部 Max 和 局部 Sum
    float local_max = -FLT_MAX;
    float local_sum = 0.0f;

    // 1. Grid-Stride Loop 或者 简单的线性扫描
    // 每个线程负责处理 idx, idx+blockDim, idx+2*blockDim ... 的数据
    for (int i = tid; i < N; i += blockDim.x) {
        float val = input_row[i];

        // 实现 Online Softmax 更新逻辑 (Thread Local)
        // 比较 local_max 和 val,更新 local_max_new
        // 利用推导公式更新 local_sum_new
        // 注意:local_sum 存储的是 exp(x - local_max) 的累加和
        float max_prev = local_max;
        local_max = fmaxf(local_max, val);
        local_sum = local_sum * expf(max_prev - local_max) + expf(val - local_max);
    }

    // --- 此时,每个线程都有了自己的 local_max 和 local_sum ---
    // 接下来需要把所有线程的结果合并成全局结果
    // 这需要一个特殊的 Reduce:不仅仅是 Sum,还要考虑 Max 的偏移

    // 2. Warp 内规约 (Warp Reduce)
    // 需要同时交换 Max 和 Sum
    // 为了简化,我们使用 Shared Memory 来做 Block Reduce

    static __shared__ float shared_max[32]; // 假设 max warps = 32
    static __shared__ float shared_sum[32];

    int lane = tid % WARP_SIZE;
    int wid = tid / WARP_SIZE;

    // --- Warp Level Online Reduce ---
    // 所有线程都有最终结果
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        float other_max = __shfl_xor_sync(0xffffffff, local_max, offset);
        float other_sum = __shfl_xor_sync(0xffffffff, local_sum, offset);

        // 合并两个线程的 (max, sum)
        // 1. global_max = max(my_max, other_max)
        // 2. global_sum = my_sum * correction + other_sum * correction

        float max_new = fmaxf(local_max, other_max);
        local_sum = local_sum * expf(local_max - max_new) +
                    other_sum * expf(other_max - max_new);
        local_max = max_new;
    }

    // Warp 结果写入 Shared Memory
    if (lane == 0) {
        shared_max[wid] = local_max;
        shared_sum[wid] = local_sum;
    }
    __syncthreads();

    // 3. Block Level Reduce (由第一个 Warp 处理)
    // 逻辑同上,只是从 Shared Memory 读
    if (wid == 0) {
        // 先把 Warp 0 自己的数据重置 (或者利用前一步的结果,这里为了逻辑清晰重置)
        // 注意:这里需要小心.简单的做法是 Warp 0 的线程读取 shared_max/sum 并再次 reduce

        local_max = (tid < blockDim.x / WARP_SIZE) ? shared_max[lane] : -FLT_MAX;
        local_sum = (tid < blockDim.x / WARP_SIZE) ? shared_sum[lane] : 0.0f;

        #pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            float other_max = __shfl_xor_sync(0xffffffff, local_max, offset);
            float other_sum = __shfl_xor_sync(0xffffffff, local_sum, offset);

            float max_new = fmaxf(local_max, other_max);
            local_sum = local_sum * expf(local_max - max_new) +
                        other_sum * expf(other_max - max_new);
            local_max = max_new;
        }

        if (tid == 0) {
            shared_max[0] = local_max;
            shared_sum[0] = local_sum;
        }
    }
    __syncthreads();

    // 4. 获取最终的全局 Max 和 Sum
    float global_max = shared_max[0];
    float global_sum = shared_sum[0];

    // 5. 第二次遍历:计算最终结果并写入
    // Softmax 本质上需要两次访问 input (第一次算 max/sum,第二次算 output)
    // 但 Online Softmax 让我们在第一次遍历时就把 max/sum 算好了,不需要中间存 temp 值

    for (int i = tid; i < N; i += blockDim.x) {
        float val = input_row[i];
        // output = e^(val - global_max) / global_sum
        output_row[i] = expf(val - global_max) / global_sum;
    }
}

// Host 包装函数
torch::Tensor launch_online_softmax(torch::Tensor input) {
    int M = input.size(0);
    int N = input.size(1);
    auto output = torch::empty_like(input);

    int block_size = 256;
    dim3 grid(M);
    dim3 block(block_size);

    softmax_online_kernel<<<grid, block>>>(
        (float*)input.data_ptr(),
        (float*)output.data_ptr(),
        N
    );
    return output;
}
"""

cpp_source = "torch::Tensor launch_online_softmax(torch::Tensor input);"

module = load_inline(
    name="online_softmax_ops",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["launch_online_softmax"],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
)

# --- 测试 ---
M, N = 10, 2048  # 可以试试更大的 N
x = torch.randn(M, N, device="cuda")

y_my = module.launch_online_softmax(x)
y_torch = torch.softmax(x, dim=1)

if torch.allclose(y_my, y_torch):
    print("✅ Online Softmax Match!")
else:
    print("❌ Mismatch!")
    # Debug info
    print("My:", y_my[0, :5])
    print("PyTorch:", y_torch[0, :5])

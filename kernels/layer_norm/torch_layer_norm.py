import torch
from torch.utils.cpp_extension import load_inline

cuda_source = """
#include <cuda_runtime.h>
#include <float.h>

#define WARP_SIZE 32

// 1. 定义统计量结构体
struct Stats {
    float s;  // sum
    float s2; // sum of squares
};

// 2. 也是 Warp Reduce,但这次要处理两个数
__device__ __forceinline__ Stats warp_reduce_stats(Stats val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        // 实现 Warp 内的 Sum 和 SumSq 归约
        float other_s = __shfl_xor_sync(0xffffffff, val.s, offset);
        float other_s2 = __shfl_xor_sync(0xffffffff, val.s2, offset);
        val.s += other_s;
        val.s2 += other_s2;
    }
    return val;
}

// 3. Block Reduce
__device__ __forceinline__ Stats block_reduce_stats(Stats val) {
    static __shared__ Stats shared[32]; // Max 32 warps
    int lane = threadIdx.x % WARP_SIZE;
    int wid = threadIdx.x / WARP_SIZE;

    // Warp Reduce
    val = warp_reduce_stats(val);

    // Write to shared
    if (lane == 0) shared[wid] = val;
    __syncthreads();

    // Collect from shared
    // 只有第一个 Warp 需要工作
    // 假设 BlockDim.x 足够大,这里简化处理
    val = (threadIdx.x < blockDim.x / WARP_SIZE) ? shared[lane] : Stats{0.0f, 0.0f};

    if (wid == 0)
        val = warp_reduce_stats(val);

    // Broadcast 结果给所有线程
    if (threadIdx.x == 0)
        shared[0] = val;
    __syncthreads();

    return shared[0];
}

__global__ void layernorm_kernel(float* x, float* out, float* gamma, float* beta, int N, float epsilon) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;

    // 定位到当前行
    float* x_row = x + bid * N;
    float* out_row = out + bid * N;

    // 1. 线程局部统计
    // 考虑到 N 可能大于 BlockDim
    Stats local_stats = {0.0f, 0.0f};

    for (int i = tid; i < N; i += blockDim.x) {
        float val = x_row[i];
        // 2: 更新局部 sum 和 sum_sq
        local_stats.s += val;
        local_stats.s2 += val * val;
    }

    // 2. Block 归约
    Stats global_stats = block_reduce_stats(local_stats);

    // 3. 计算 Mean 和 Variance
    // 利用公式计算 mu 和 sigma (rsigma = 1 / sqrt(var + eps))
    //方差公式 Var = E[x^2] - (E[x])^2
    float mu = global_stats.s / N;
    float var = (global_stats.s2 / N) - (mu * mu);
    float rsigma = rsqrtf(var + epsilon);

    // 4. 归一化并写入
    for (int i = tid; i < N; i += blockDim.x) {
        float val = x_row[i];
        // TODO 4: 计算 LayerNorm 最终结果
        // out = (x - mu) * rsigma * gamma + beta
        // 记得 gamma 和 beta 也是向量
        float norm_val = (val - mu) * rsigma;
        out_row[i] = norm_val * gamma[i] + beta[i];
    }
}

torch::Tensor launch_layernorm(torch::Tensor x, torch::Tensor gamma, torch::Tensor beta, float eps) {
    int M = x.size(0); // Batch Size
    int N = x.size(1); // Hidden Size
    auto output = torch::empty_like(x);

    dim3 grid(M);
    dim3 block(std::min(N, 1024)); // 简单的 Block 配置

    layernorm_kernel<<<grid, block>>>(
        (float*)x.data_ptr(),
        (float*)output.data_ptr(),
        (float*)gamma.data_ptr(),
        (float*)beta.data_ptr(),
        N,
        eps
    );
    return output;
}
"""

cpp_source = "torch::Tensor launch_layernorm(torch::Tensor x, torch::Tensor gamma, torch::Tensor beta, float eps);"

module = load_inline(
    name="layernorm_ops",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["launch_layernorm"],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
)

# --- 测试 ---
M, N = 128, 1024
x = torch.randn(M, N, device="cuda")
gamma = torch.ones(N, device="cuda")
beta = torch.zeros(N, device="cuda")
eps = 1e-5

y_my = module.launch_layernorm(x, gamma, beta, eps)
y_torch = torch.nn.functional.layer_norm(x, (N,), weight=gamma, bias=beta, eps=eps)

if torch.allclose(y_my, y_torch, atol=1e-5):
    print("✅ LayerNorm Match!")
else:
    print("❌ Mismatch!")
    print("My:", y_my[0, :5])
    print("PyTorch:", y_torch[0, :5])

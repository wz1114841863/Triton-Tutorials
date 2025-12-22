import torch
from torch.utils.cpp_extension import load_inline

cuda_source = """
#include <cuda_runtime.h>
#include <float.h>

#define WARP_SIZE 32

// Warp Reduce Sum
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
    return val;
}

// Block Reduce Sum
__device__ __forceinline__ float block_reduce_sum(float val) {
    static __shared__ float shared[32];
    int lane = threadIdx.x % WARP_SIZE;
    int wid = threadIdx.x / WARP_SIZE;

    val = warp_reduce_sum(val);
    if (lane == 0) shared[wid] = val;
    __syncthreads();

    val = (threadIdx.x < blockDim.x / WARP_SIZE) ? shared[lane] : 0.0f;
    if (wid == 0) val = warp_reduce_sum(val);

    if (threadIdx.x == 0) shared[0] = val;
    __syncthreads();

    return shared[0];
}

__global__ void rmsnorm_kernel(float* x, float* out, float* weight, int N, float epsilon) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;

    // 定位到当前行
    float* x_row = x + bid * N;
    float* out_row = out + bid * N;

    // 1. 线程局部求平方和
    float local_sum_sq = 0.0f;
    for (int i = tid; i < N; i += blockDim.x) {
        float val = x_row[i];
        // 累加平方和
        local_sum_sq += val * val;
    }

    // 2. Block 归约得到全局平方和
    float global_sum_sq = block_reduce_sum(local_sum_sq);

    // 3. 计算 RMS 的倒数 (Reciprocal RMS)
    // rms = sqrt(sum_sq / N + eps)
    // r_rms = 1 / rms = rsqrt(sum_sq / N + eps)
    float r_rms = rsqrtf(global_sum_sq / N + epsilon);

    // 4. 归一化并缩放
    for (int i = tid; i < N; i += blockDim.x) {
        float val = x_row[i];
        // 计算最终结果
        // out = val * r_rms * weight
        out_row[i] = val * r_rms * weight[i];
    }
}

torch::Tensor launch_rmsnorm(torch::Tensor x, torch::Tensor weight, float eps) {
    int M = x.size(0);
    int N = x.size(1);
    auto out = torch::empty_like(x);

    dim3 grid(M);
    dim3 block(std::min(N, 1024));

    rmsnorm_kernel<<<grid, block>>>(
        (float*)x.data_ptr(),
        (float*)out.data_ptr(),
        (float*)weight.data_ptr(),
        N,
        eps
    );
    return out;
}
"""

cpp_source = (
    "torch::Tensor launch_rmsnorm(torch::Tensor x, torch::Tensor weight, float eps);"
)

module = load_inline(
    name="rmsnorm_ops",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["launch_rmsnorm"],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
)


# --- 测试 ---
class RMSNorm(torch.nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x):
        input_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x.to(input_dtype)


M, N = 128, 1024
x = torch.randn(M, N, device="cuda")
weight = torch.ones(N, device="cuda")  # LLaMA 默认初始化为 1
eps = 1e-6

# 运行自定义算子
y_my = module.launch_rmsnorm(x, weight, eps)

# 运行 PyTorch 基准 (手动实现,因为 torch.nn.RMSNorm 是比较新的版本才有的)
rms_ref = RMSNorm(N, eps=eps).cuda()
rms_ref.weight.data = weight
y_ref = rms_ref(x)

if torch.allclose(y_my, y_ref, atol=1e-5):
    print("✅ RMSNorm Match!")
else:
    print("❌ Mismatch!")
    print("My:", y_my[0, :5])
    print("Ref:", y_ref[0, :5])

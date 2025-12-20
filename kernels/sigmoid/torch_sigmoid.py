import torch
from torch.utils.cpp_extension import load_inline

cuda_source = """
#include <cuda_runtime.h>

// 定义 Sigmoid 仿函数
struct SigmoidFunctor {
    __device__ __forceinline__ float operator()(float x) const {
        return 1.0f / (1.0f + expf(-x));
    }
};

// 定义 Tanh 仿函数
struct TanhFunctor {
    __device__ __forceinline__ float operator()(float x) const {
        return tanhf(x);
    }
};

template <typename Func>
__global__ void element_unary_op_kernel(const float* x, float* y, int n, Func func) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // float4 优化部分
    int n_vec = n / 4;
    const float4* x_vec = reinterpret_cast<const float4*>(x);
    float4* y_vec = reinterpret_cast<float4*>(y);

    while (idx < n_vec) {
        float4 vx = x_vec[idx];
        float4 vy;

        vy.x = func(vx.x);
        vy.y = func(vx.y);
        vy.z = func(vx.z);
        vy.w = func(vx.w);

        y_vec[idx] = vy;
        idx += stride;
    }

    // 尾部处理 (Scalar Loop)
    for (int j = n_vec * 4 + (blockIdx.x * blockDim.x + threadIdx.x);
             j < n;
             j += stride) {
        y[j] = func(x[j]);
    }
}

torch::Tensor launch_sigmoid(torch::Tensor x) {
    auto y = torch::empty_like(x);
    int n = x.numel();
    int block_size = 256;
    int num_sms = 30;
    int grid_size = num_sms * 4; // 保持足够的并行度

    element_unary_op_kernel<<<grid_size, block_size>>>(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        n,
        SigmoidFunctor()
    );
    return y;
}

torch::Tensor launch_tanh(torch::Tensor x) {
    auto y = torch::empty_like(x);
    int n = x.numel();
    int block_size = 256;
    int num_sms = 30;
    int grid_size = num_sms * 4;

    element_unary_op_kernel<<<grid_size, block_size>>>(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        n,
        TanhFunctor()
    );
    return y;
}
"""

cpp_source = """
torch::Tensor launch_sigmoid(torch::Tensor x);
torch::Tensor launch_tanh(torch::Tensor x);
"""

module = load_inline(
    name="element_unary_ops",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["launch_sigmoid", "launch_tanh"],
    # 使用硬件内置的快速近似指令(Intrinsic Functions)来替代标准的数学函数.
    # 把 expf (x) 自动替换成 __expf(x)
    extra_cuda_cflags=["-O3", "--use_fast_math"],
)

# --- 测试 ---
n = 1024 * 1024 * 10 + 3
x = torch.randn(n, device="cuda")

# Sigmoid 测试
y_my = module.launch_sigmoid(x)
y_torch = torch.sigmoid(x)
if torch.allclose(y_my, y_torch):
    print("✅ Sigmoid Match!")
else:
    print("❌ Sigmoid Mismatch! (Check logic)")

# Tanh 测试
y_tanh_my = module.launch_tanh(x)
y_tanh_torch = torch.tanh(x)
if torch.allclose(y_tanh_my, y_tanh_torch):
    print("✅ Tanh Match!")

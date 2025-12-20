import torch
from torch.utils.cpp_extension import load_inline


def get_optimal_grid_size(n, block_size=256):
    # 获取显卡的 SM 数量
    num_sms = torch.cuda.get_device_properties(0).multi_processor_count
    # 设定目标 Grid Size
    # 经验值:SM 数量 * 每个 SM 驻留的 Block 数 (通常取 32 左右作为上限足够了)
    target_grid_size = num_sms * 32
    # 边界处理: 如果数据量 N 很小,没必要启动几千个 Block
    needed_blocks = (n + block_size - 1) // block_size
    # 4. 取两者的最小值
    grid_size = min(target_grid_size, needed_blocks)
    return grid_size


cuda_source = """
#include <cuda_runtime.h>

#define FLOAT4(value) (reinterpret_cast<float4*>(&(value))[0])

struct ReLUFunctor {
    __device__ __forceinline__ float operator()(float a, float b) const {
        // z = max(x + y, 0)
        float val = a + b;
        return val > 0.0f ? val : 0.0f;
    }
};

template <typename Func>
__global__ void element_ops_kernel(float* x, float* y, float* z, int n, Func func) {
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // float4 的索引
    int stride = blockDim.x * gridDim.x;

    // 1. 向量化部分 (Vectorized Loop)
    // 假设 n 是 float 的总数,那么 float4 的总数是 n / 4
    int n_vec = n / 4;

    while (idx < n_vec) {
        // 使用 float4 读取 x 和 y
        float4 v_x = FLOAT4(x[idx * 4]);
        float4 v_y = FLOAT4(y[idx * 4]);

        float4 v_z;
        // 2: 执行加法 (注意 float4 不能直接相加,要分量 x,y,z,w 分别相加)
        // v_z.x = v_x.x + v_y.x;
        // v_z.y = v_x.y + v_y.y;
        // v_z.z = v_x.z + v_y.z;
        // v_z.w = v_x.w + v_y.w;
        v_z.x = func(v_x.x, v_y.x);
        v_z.y = func(v_x.y, v_y.y);
        v_z.z = func(v_x.z, v_y.z);
        v_z.w = func(v_x.w, v_y.w);

        // 3. 将结果写回 z (也是用 float4)
        FLOAT4(z[idx * 4]) = v_z; // 提示:这里的索引要注意,如果是指针强转方式,直接用 input_v[idx] 风格更方便

        idx += stride;
    }

    // 2. 尾部处理 (Scalar Loop)
    // 处理剩下的 1~3 个元素
    int i = n_vec * 4 + tid;
    while (i < n) {
        z[i] = func(x[i], y[i]);
        i += stride;
    }
}

torch::Tensor launch_relu(torch::Tensor x, torch::Tensor y) {
    auto z = torch::empty_like(x);
    int n = x.numel();
    int block_size = 256;
    int num_sms = 30;
    int grid_size = num_sms * 4;

    // 实例化:装上 AddFunctor
    element_ops_kernel<<<grid_size, block_size>>>(
        x.data_ptr<float>(), y.data_ptr<float>(), z.data_ptr<float>(), n, ReLUFunctor()
    );
    return z;
}
"""

cpp_source = """
    torch::Tensor launch_add(torch::Tensor x, torch::Tensor y);
    torch::Tensor launch_mul(torch::Tensor x, torch::Tensor y);
"""

module = load_inline(
    name="element_ops",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["launch_add", "launch_mul"],
    extra_cuda_cflags=["-O3"],
    verbose=True,
)

# 测试
n = 1024 * 1024 + 3
x = torch.randn(n, device="cuda")
y = torch.randn(n, device="cuda")

# 测试ReLU
z_relu = module.launch_relu(x, y)
assert torch.allclose(z_relu, torch.relu(x + y))
print("ReLU Match!")

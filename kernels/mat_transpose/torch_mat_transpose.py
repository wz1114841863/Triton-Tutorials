import torch
from torch.utils.cpp_extension import load_inline

cuda_source = """
#include <cuda_runtime.h>

// Tile Dimension (通常设为 32)
#define TILE_DIM 32

__global__ void transpose_kernel(const float* idata, float* odata, int width, int height) {
    // 1. 定义 Shared Memory
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];

    // 2. 计算 "读" 阶段的坐标
    // (xIndex, yIndex) 是输入矩阵中的坐标
    int xIndex = blockIdx.x * TILE_DIM + threadIdx.x;
    int yIndex = blockIdx.y * TILE_DIM + threadIdx.y;

    // 输入矩阵的线性索引 (行优先: y * width + x)
    int index_in = yIndex * width + xIndex;

    // 读取数据到 Shared Memory
    // 边界检查:确保 xIndex < width && yIndex < height
    if (xIndex < width && yIndex < height) {
        tile[threadIdx.y][threadIdx.x] = idata[index_in];
    }

    // 同步
    __syncthreads();

    // 3. 计算 "写" 阶段的坐标 (关键逻辑!)
    // 我们想要 "合并写入" (Coalesced Write).
    // 这意味着:threadIdx.x (0, 1, 2...) 应该对应 odata 内存中连续的地址.
    // odata 是转置后的矩阵,它的 "行" 是原来的 "列".
    // 所以,我们需要重新计算 x 和 y,使得 threadIdx.x 依然对应输出矩阵的 "列索引" (连续变化).

    //原来的 blockIdx.y (控制行) 现在应该控制输出的列 -> 变成输出的 x 坐标的一部分
    //原来的 blockIdx.x (控制列) 现在应该控制输出的行 -> 变成输出的 y 坐标的一部分
    // 交换 Block 的角色,但不交换 Thread 的角色, 并不直接对应转置
    int xIndex_new = blockIdx.y * TILE_DIM + threadIdx.x;
    int yIndex_new = blockIdx.x * TILE_DIM + threadIdx.y;

    int index_out = yIndex_new * height + xIndex_new; // height 是输出矩阵的 width

    // 从 Shared Memory 写出到 Global Memory
    // 我们刚才读入的时候是 tile[ty][tx] = input(x, y)
    // 现在我们要写出 output(x_new, y_new),这对应原来的 input(y_new, x_new)
    if (xIndex_new < height && yIndex_new < width) {
        odata[index_out] = tile[threadIdx.x][threadIdx.y];
    }
}

torch::Tensor launch_transpose(torch::Tensor x) {
    int height = x.size(0);
    int width = x.size(1);

    auto y = torch::zeros({width, height}, x.options());

    // Grid 配置
    dim3 block(TILE_DIM, TILE_DIM);
    dim3 grid((width + TILE_DIM - 1) / TILE_DIM, (height + TILE_DIM - 1) / TILE_DIM);

    transpose_kernel<<<grid, block>>>(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        width,
        height
    );

    return y;
}
"""

cpp_source = "torch::Tensor launch_transpose(torch::Tensor x);"

module = load_inline(
    name="transpose_ops",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["launch_transpose"],
    extra_cuda_cflags=["-O3"],
)

# --- 测试 ---
H, W = 4096, 4096
x = torch.randn(H, W, device="cuda")

# 运行自定义算子
y_my = module.launch_transpose(x)
# 运行 PyTorch 原生算子
y_torch = x.t()  # 或者 torch.transpose(x, 0, 1)

if torch.allclose(y_my, y_torch):
    print("✅ Transpose Match!")
else:
    print("❌ Mismatch!")

import torch
from torch.utils.cpp_extension import load_inline

cuda_source = """
#include <cuda_runtime.h>

#define FLOAT4(value) (reinterpret_cast<float4*>(&(value))[0])

__global__ void embedding_forward_kernel(const int* indices, const float* weight, float* output, int D, int padding_idx) {
    // 策略:Grid 的每一个 Block 处理 1 个 Token (即 Output 的 1 行)
    // blockIdx.x: 当前处理的是第几个 Token (sequence dimension)
    unsigned int token_idx = blockIdx.x;

    // 1. 获取当前 Token 在词表中的 ID
    unsigned int vocab_idx = indices[token_idx];

    // 2. 处理 Padding (如果遇到 padding_idx,输出全 0)
    if (vocab_idx == padding_idx) {
        // 假设output初始化是0
        return;
    }

    // 3. 计算这一行在 Weight 中的起始地址
    // weight shape: [V, D]
    const float* src_row = weight + vocab_idx * D;

    // 4. 计算 Output 的起始地址
    // output shape: [N, D]
    float* dst_row = output + token_idx * D;

    // 5. 向量化搬运 (Collaborative Copy)
    // Block 内的线程合作搬运长度为 D 的一行数据
    int tid = threadIdx.x;
    int stride = blockDim.x * 4; // 每次搬运 float4

    // 强转指针
    const float4* src_vec = reinterpret_cast<const float4*>(src_row);
    float4* dst_vec = reinterpret_cast<float4*>(dst_row);

    // 向量化循环
    int n_vec = D / 4;  // 这里是整数除法, 忽略尾部
    for (int i = tid; i < n_vec; i += blockDim.x) {
        // 搬运数据
        dst_vec[i] = src_vec[i];
    }

    // 6. 尾部处理 (处理 D 不能被 4 整除的情况)
    int rem_start = n_vec * 4;
    for (int i = rem_start + tid; i < D; i += blockDim.x) {
        dst_row[i] = src_row[i];
    }
}

torch::Tensor launch_embedding(torch::Tensor indices, torch::Tensor weight, int padding_idx = -1) {
    int N = indices.numel(); // Token 总数
    int V = weight.size(0);  // 词表大小
    int D = weight.size(1);  // Embedding 维度

    auto output = torch::empty({N, D}, weight.options());

    // 策略:有多少个 Token 就启动多少个 Block
    // 注意:如果 N 很大超过 65535,早期的 GPU 可能受限,但现代 GPU Grid x 维可以很大 (2^31-1)
    int grid_size = N;
    int block_size = 512; // D一般很大, 可以设大一点, 比如 512

    embedding_forward_kernel<<<grid_size, block_size>>>(
        indices.data_ptr<int>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        D,
        padding_idx
    );

    return output;
}
"""

cpp_source = "torch::Tensor launch_embedding(torch::Tensor indices, torch::Tensor weight, int padding_idx);"

module = load_inline(
    name="embedding_ops",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["launch_embedding"],
    extra_cuda_cflags=["-O3"],
)

# --- 测试 ---
V, D = 1000, 1024
N = 100  # Batch size * Seq len
weight = torch.randn(V, D, device="cuda")
# 生成随机索引
indices = torch.randint(0, V, (N,), device="cuda", dtype=torch.int32)

# 运行自定义算子
out_my = module.launch_embedding(indices, weight, -1)
# 运行 PyTorch 原生算子
out_torch = torch.nn.functional.embedding(indices.long(), weight)

if torch.allclose(out_my, out_torch):
    print("✅ Embedding Match!")
else:
    print("❌ Mismatch!")

import torch
from torch.utils.cpp_extension import load_inline

cuda_source = """
#include <cuda_runtime.h>
#include <float.h>
#include <math.h>

// 假设 N 是序列长度,d 是 head 维度 (64)
// Br = 64 (Q 的分块大小), Bc = 64 (K/V 的分块大小)
#define Br 64
#define Bc 64
#define d 64

__global__ void flash_attention_kernel(
    float* Q, float* K, float* V, float* O, float* L,
    int N, float softmax_scale)
{
    // 1. 确定当前 Thread 负责 Q 的哪一行
    // Grid 维度通常是 (N / Br), 每个 Block 负责 Q 的 [tx_start, tx_end) 行
    int tx = threadIdx.x; // 线程在 Block 内的索引
    int row_idx = blockIdx.x * Br + tx; // 全局行号

    // 边界检查:防止越界 (假设 N 可能不是 Br 的倍数)
    if (row_idx >= N) return;

    // 2. 初始化寄存器 (Accumulators)
    // O_row: 存储当前行的输出结果 (长度为 d)
    float O_row[d] = {0.0f};
    float l = 0.0f;  // running sum
    float m = -INFINITY; // running max

    // 3. 加载Q blockShared Memory,
    // 采用float4 类型加载, 默认d可以整除4
    __shared__ float Q_shared[Br][d];
    for (int t = 0; t < d; t += 4) {
        // 每个线程加载自己负责的Q行
        float4 q_vec = *reinterpret_cast<float4*>(&Q[row_idx * d + t]);
        *reinterpret_cast<float4*>(&Q_shared[tx][t]) = q_vec;
    }

    // Shared Memory 用于缓存 K 和 V 的一个 Block
    __shared__ float K_shared[Bc][d];
    __shared__ float V_shared[Bc][d];

    __syncthreads();  // 确保 Q 已加载完成

    // 4. 主循环:遍历 K 和 V 的所有块Tc (Loop over KV blocks)
    for (int j = 0; j < N; j += Bc) {
        // --- 阶段 A: 加载 K, V 到 Shared Memory ---
        // 线程块的大小为Br = M/4d, Bc = min(M/4d, d), Bc < Br
        int k_row_global = j + tx;
        if (k_row_global < N) { // K, V 的边界检查
            for (int t = 0; t < d; t += 4) {
                float4 k_vec = *reinterpret_cast<float4*>(&K[k_row_global * d + t]);
                float4 v_vec = *reinterpret_cast<float4*>(&V[k_row_global * d + t]);
                *reinterpret_cast<float4*>(&K_shared[tx][t]) = k_vec;
                *reinterpret_cast<float4*>(&V_shared[tx][t]) = v_vec;
            }
        } else {
             // 越界部分填 0,防止计算出错 (Padding)
             for (int t = 0; t < d; t += 4) {
                float4 zero_vec = {0.0f, 0.0f, 0.0f, 0.0f};
                *reinterpret_cast<float4*>(&K_shared[tx][t]) = zero_vec;
                *reinterpret_cast<float4*>(&V_shared[tx][t]) = zero_vec;
             }
        }
        __syncthreads(); // 等待 K/V 加载完成

        // --- 阶段 B: 计算 S = Q * K^T ---
        // 当前线程计算 Q_row 与 K_shared 所有列的点积
        float S[Bc]; // 存储当前行与 K Block 中每一列的 Attention Score
        for (int k = 0; k < Bc; ++k) {
            float score = 0.0f;
            for (int i = 0; i < d; ++i) {
                score += Q_shared[tx][i] * K_shared[k][i];
            }
            // 应用缩放因子, 1/ sqrt(d)
            S[k] = score * softmax_scale;
        }

        // --- 阶段 C: Online Softmax 更新逻辑 (核心!) ---

        // 1. 找出当前 Block 的最大值
        float m_block = -INFINITY;
        for (int k = 0; k < Bc; ++k) {
            m_block = fmaxf(m_block, S[k]);
        }

        // 2. 更新全局最大值 m_new
        float m_prev = m;
        float m_new = fmaxf(m, m_block);

        // 3. 计算修正因子
        // 如果 m 变大了,之前的累加值需要缩小
        float scale_prev = expf(m - m_new);
        float scale_curr = expf(m_block - m_new);

        // 4. 更新 P (Attention Probability, unnormalized)
        float P[Bc];
        float l_block = 0.0f;
        for (int k = 0; k < Bc; ++k) {
            P[k] = expf(S[k] - m_new);
            l_block += P[k];
        }

        // 5. 更新 l (running sum)
        l = l * scale_prev + l_block;

        // 5. 更新 O (Output Accumulator)
        // O_new = O_prev * scale_prev + P * V
        for (int i = 0; i < d; ++i) {
            // 先把旧的 O 缩放
            float o_val = O_row[i] * scale_prev;

            // 累加当前块的 P * V
            float pv_sum = 0.0f;
            for (int k = 0; k < Bc; ++k) {
                pv_sum += P[k] * V_shared[k][i];
            }
            O_row[i] = o_val + pv_sum;
        }

        // 6. 更新m (running max)
        m = m_new;

        __syncthreads(); // 准备加载下一块
    }

    // 5. 最终归一化并写回 Global Memory
    // 写入 O
    for (int i = 0; i < d; i+=4) {
        float4 o_vec;
        o_vec.x = O_row[i] / l;
        o_vec.y = O_row[i+1] / l;
        o_vec.z = O_row[i+2] / l;
        o_vec.w = O_row[i+3] / l;
        *reinterpret_cast<float4*>(&O[row_idx * d + i]) = o_vec;
    }

    // 写入 L (供 Backward 使用)
    // L = m + log(l)
    if (L != nullptr) {
        L[row_idx] = m + logf(l);
    }
}

torch::Tensor Launch_flash_attention_v2(torch::Tensor Q, torch::Tensor K, torch::Tensor V, float softmax_scale) {
    const int N = Q.size(0); // 序列长度
    auto O = torch::zeros_like(Q); // 输出张量
    auto L = torch::zeros({N}, Q.options()); // 用于存储 L
    const int threads = Br; // 每个 Block 的线程数
    const int blocks = (N + Br - 1) / Br; // 计算 Block 数量

    flash_attention_kernel<<<blocks, threads>>>(
        (float*)Q.data_ptr(),
        (float*)K.data_ptr(),
        (float*)V.data_ptr(),
        (float*)O.data_ptr(),
        (float*)L.data_ptr(),
        N,
        softmax_scale
    );
    return O;
}
"""

cpp_source = "torch::Tensor Launch_flash_attention_v2(torch::Tensor Q, torch::Tensor K, torch::Tensor V, float softmax_scale);"

module = load_inline(
    name="flash_attention_v2_ops",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["Launch_flash_attention_v2"],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
)

# 测试数据
d = 64
N = 128
# 确保数据是连续的,且在 GPU 上
Q = torch.randn(N, d).cuda().contiguous()
K = torch.randn(N, d).cuda().contiguous()
V = torch.randn(N, d).cuda().contiguous()
scale = 1.0 / (d**0.5)

O_torch = torch.nn.functional.scaled_dot_product_attention(Q, K, V, scale=scale)
O_my = module.Launch_flash_attention_v2(Q, K, V, scale)

# 验证结果
diff = torch.max(torch.abs(O_torch - O_my)).item()
print(f"Max difference: {diff}")

if diff < 1e-3:
    print("✅ 验证通过!结果一致.")
else:
    print("❌ 差异过大,请检查逻辑.")

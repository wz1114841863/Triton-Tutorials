import torch
from torch.utils.cpp_extension import load_inline

cuda_source = """
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define WARP_SIZE 32
#define LDST128BITS(value) (reinterpret_cast<float4 *>(&(value))[0])

// warp reduce sum for float
__device__ __forceinline__ float warp_reduce_sum_f32(float val) {
    #pragma unroll
    for (int mask = WARP_SIZE >> 1; mask >= 1; mask >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, mask);
    }
    return val;
}

// 假设 N 是 8 的倍数,输入是 half
template <const int NUM_THREADS = 256>
__global__ void dot_product_f16x8_pack_f32_kernel(half *a, half *b, float *result, int N) {
    int tid = threadIdx.x;
    // 每个线程处理 8 个元素
    int idx = (blockIdx.x * NUM_THREADS + tid) * 8;

    // 1. 准备 Shared Memory (用于 Block 级归约)
    constexpr int NUM_WARPS = NUM_THREADS / WARP_SIZE;
    __shared__ float smem[NUM_WARPS];

    // 2. 向量化加载 (Vectorized Load)
    // 既然是 Dot Product,就要同时读 a 和 b
    half pack_a[8];
    half pack_b[8];

    // 初始化局部累加器为 FP32 (防溢出关键!)
    float sum_f32 = 0.0f;

    if (idx < N) { // 简单边界检查
        // 利用 float4 加载 128-bit
        LDST128BITS(pack_a[0]) = LDST128BITS(a[idx]);
        LDST128BITS(pack_b[0]) = LDST128BITS(b[idx]);

        // 3. 计算
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            // A: 激进防溢出 (转 FP32 再乘) -> 精度最高,稍慢
            float va = __half2float(pack_a[i]);
            float vb = __half2float(pack_b[i]);
            sum_f32 += va * vb;

            // B: 适度防溢出 (FP16 乘,FP32 加) -> 速度快,通常够用
            // 只有当 a*b 单次乘积能超过 65504 时才需要策略 A
            // sum_f32 += __half2float(pack_a[i] * pack_b[i]);
        }
    }

    // 4. Warp Reduce
    sum_f32 = warp_reduce_sum_f32(sum_f32);

    // 5. Block Reduce (通过 Shared Memory)
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;

    if (lane_id == 0) smem[warp_id] = sum_f32;
    __syncthreads();

    // 让 Warp 0 里的前几个线程把 smem 里的值加起来
    sum_f32 = (lane_id < NUM_WARPS) ? smem[lane_id] : 0.0f;

    if (warp_id == 0) {
        sum_f32 = warp_reduce_sum_f32(sum_f32);
    }

    // 6. Global Reduce (Atomic Add)
    if (tid == 0) {
        atomicAdd(result, sum_f32);
    }
}

torch::Tensor launch_dot(torch::Tensor a, torch::Tensor b) {
    int N = a.numel();
    auto result = torch::zeros({}, a.options().dtype(torch::kFloat));

    int block_size = 256;
    // 每个线程处理 8 个,每个 Block 处理 256*8 = 2048 个
    int grid_size = (N + 2048 - 1) / 2048;

    dot_product_f16x8_pack_f32_kernel<256><<<grid_size, block_size>>>(
        (half*)a.data_ptr(),
        (half*)b.data_ptr(),
        (float*)result.data_ptr(),
        N
    );
    return result;
}
"""

cpp_source = "torch::Tensor launch_dot(torch::Tensor a, torch::Tensor b);"

module = load_inline(
    name="dot_product_ops",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["launch_dot"],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
)

# --- 测试 ---
N = 1024 * 1024
# 构造容易溢出的数据: 100 * 100 = 10000, 累加 7 个就溢出 FP16
a = torch.full((N,), 100.0, device="cuda", dtype=torch.half)
b = torch.full((N,), 100.0, device="cuda", dtype=torch.half)

res_my = module.launch_dot(a, b)
res_torch = torch.dot(a.float(), b.float())  # 用 float 算基准

print(f"My Dot: {res_my.item()}")
print(f"Torch Dot: {res_torch.item()}")

if torch.allclose(res_my, res_torch):
    print("✅ Dot Product Match!")
else:
    print("❌ Mismatch!")

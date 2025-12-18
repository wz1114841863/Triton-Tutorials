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


# CUDA 内核代码实现
cuda_source = """
__global__ void reduce_sum_kernel_v1(const float* input, float* output, int n) {
    // 1. 计算全局索引
    int tid = threadIdx.x;

    // 声明 Shared Memory
    __shared__ float sdata[256];  // 假设 blockDim.x == 256

    // 2. 将数据从 Global Memory 加载到 Shared Memory
    sdata[tid] = (tid < n) ? input[tid] : 0.0f;

    // 3. 同步, 等待所有线程加载完成
    __syncthreads();

    // 4. 实现树形归约 (Tree Reduction)
    for (unsigned int j = blockDim.x / 2; j > 0; j >>= 1) {
        if (tid < j) {
            sdata[tid] += sdata[tid + j];
        }
        __syncthreads();
    }

    // 5. 将结果写回 Global Memory
    if (tid == 0) {
        *output = sdata[0];
    }
}

__global__ void reduce_sum_kernel_v2(const float* input, float* output, int n) {
    // 1. 计算全局索引
    unsigned int tid = threadIdx.x; // 局部索引:0 ~ 255 (用于 Shared Memory)
    // unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; // 全局索引:0 ~ N (用于 Input)

    // 声明 Shared Memory
    __shared__ float sdata[256];

    // 2. 将数据从 Global Memory 加载到 Shared Memory
    sdata[tid] = (tid < n) ? input[tid] : 0.0f;

    // 3. 同步, 等待所有线程加载完成
    __syncthreads();

    // 4. 实现树形归约 (Tree Reduction)
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // 5. 将结果写回 Global Memory
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

__global__ void reduce_sum_kernel_v3(const float* input, float* output, int n) {
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    // 声明 Shared Memory
    __shared__ float sdata[256];
    sdata[tid] = 0.0f;

    // 将数据从 Global Memory 加载到 Shared Memory
    unsigned int grid_size = blockDim.x * gridDim.x;
    while (i < n) {
        sdata[tid] += input[i];
        i += grid_size;
    }

    __syncthreads();

    // 实现树形归约 (Tree Reduction)
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // 将结果写回 Global Memory
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

__global__ void reduce_sum_kernel_v4(const float* input, float* output, int n) {
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    // 声明 Shared Memory
    __shared__ float sdata[256];
    sdata[tid] = 0.0f;

    // 将数据从 Global Memory 加载到 Shared Memory
    unsigned int grid_size = blockDim.x * gridDim.x;
    while (i < n) {
        sdata[tid] += input[i];
        i += grid_size;
    }

    __syncthreads();

    // 实现树形归约 (Tree Reduction)
    // 这里的s指代有多少线程在参与归约, s=32时剩余的有效数据为sdata[0]~sdata[63]
    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // 最后 warp 内归约

    // 错误: 编译器会进行优化,导致缓存不一致
    // for (unsigned int offset = 32; offset > 0; offset /= 2) {
    //     if (tid < offset) {
    //         sdata[tid] += sdata[tid + offset];
    //     }
    // }

    // 正确: 手动展开, 使用 volatile 指针
    // 但是还能进一步优化, 直接访问寄存器, 不再访问 Shared Memory
    if (tid < 32) {
        // 使用 volatile 指针, 禁止优化
        // 但是依旧在使用 Shared Memory
        volatile float* vsmem = sdata;
        // 这里的逻辑必须手动展开, 不仅是为了速度, 也是为了防止编译器乱优化
        vsmem[tid] += vsmem[tid + 32];
        vsmem[tid] += vsmem[tid + 16];
        vsmem[tid] += vsmem[tid + 8];
        vsmem[tid] += vsmem[tid + 4];
        vsmem[tid] += vsmem[tid + 2];
        vsmem[tid] += vsmem[tid + 1];
    }

    // 将结果写回 Global Memory
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}


__device__ float warpReduceSum(float val) {
    // 这里的 mask 0xffffffff 表示参与的线程掩码,通常全选
    // 原理:直接从寄存器拿邻居的数据,不需要 sdata

    // 偏移 16:  0+16, 1+17...
    val += __shfl_down_sync(0xffffffff, val, 16);
    // 偏移 8:   0+8,  1+9...
    val += __shfl_down_sync(0xffffffff, val, 8);
    // 偏移 4...
    val += __shfl_down_sync(0xffffffff, val, 4);
    val += __shfl_down_sync(0xffffffff, val, 2);
    val += __shfl_down_sync(0xffffffff, val, 1);

    // 最后,线程0的val里存的就是这32个线程的总和
    return val;
}

__global__ void reduce_sum_kernel_v5(const float* input, float* output, int n) {
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    // 声明 Shared Memory
    __shared__ float sdata[256];
    sdata[tid] = 0.0f;

    // 将数据从 Global Memory 加载到 Shared Memory
    unsigned int grid_size = blockDim.x * gridDim.x;
    while (i < n) {
        sdata[tid] += input[i];
        i += grid_size;
    }

    __syncthreads();

    // 实现树形归约 (Tree Reduction)
    // 这里的s指代有多少线程在参与归约, s=32时剩余的有效数据为sdata[0]~sdata[63]
    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // 最后 warp 内归约

    // 错误: 编译器会进行优化,导致缓存不一致
    // for (unsigned int offset = 32; offset > 0; offset /= 2) {
    //     if (tid < offset) {
    //         sdata[tid] += sdata[tid + offset];
    //     }
    // }

    // 正确: 手动展开, 使用 volatile 指针
    // 但是还能进一步优化, 直接访问寄存器, 不再访问 Shared Memory
    if (tid < 32) {
        // 从 Shared Memory 拿出来,放回寄存器
        float val = sdata[tid];
        // 看起来是多做了一次访问 Shared Memory, 但实际上编译器会优化掉
        // 指令级并行
        if (blockDim.x >= 64) val += sdata[tid + 32];
        // 使用 Warp Shuffle 极速归约
        val = warpReduceSum(val);
        // 只有线程 0 写回结果
        if (tid == 0) {
            output[blockIdx.x] = val;
        }
    }
}


__global__ void reduce_sum_kernel_v6(const float* input, float* output, int n) {
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    // 强转指针: 让input变成float4类型的指针
    const float4* input_v = reinterpret_cast<const float4*>(input);
    unsigned int n_vec = n / 4; // 计算有多少个float4

    float sum = 0.0f;
    int grid_size = blockDim.x * gridDim.x;

    // 向量化循环
    while (i < n_vec) {
        float4 val = input_v[i];  // 一条指令加载 128 bit
        sum += val.x + val.y + val.z + val.w;  // 纯寄存器加法，极快
        i += grid_size;
    }

    // 处理剩余的元素
    // 如果 n 不是 4 的倍数,比如 n=1001,最后还剩 1 个数没加
    // 所有的线程都要去检查一下原数组的最后部分
    int idx_scalar = n_vec * 4 + tid; // 切换回标量索引
    while (idx_scalar < n) {
        sum += input[idx_scalar];
        idx_scalar += grid_size;
    }

    // 声明 Shared Memory
    static __shared__ float sdata[256];
    sdata[tid] = sum;
    __syncthreads();

    __syncthreads();

    // 实现树形归约 (Tree Reduction)
    // 这里的s指代有多少线程在参与归约, s=32时剩余的有效数据为sdata[0]~sdata[63]
    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // 最后 warp 内归约
    if (tid < 32) {
        // 从 Shared Memory 加载当前的部分和
        float val = sdata[tid];

        if (blockDim.x > 32) {
             val += sdata[tid + 32];
        }

        // 极速 Warp Shuffle
        val = warpReduceSum(val);

        // 写回结果
        if (tid == 0) {
            output[blockIdx.x] = val;
        }
    }
}

torch::Tensor reduce_sum(torch::Tensor input) {
    auto n = input.numel();
    auto current_input = input;

    // 设定 Block 大小
    const int block_size = 256;

    // 循环归约
    while (n > 1) {
        // 计算这一轮需要的 Grid Size
        // 如果数据量很大,Grid Size 会很大;如果数据量小,Grid Size 就小
        unsigned int grid_size = (n + block_size - 1) / block_size;

        // 准备输出 Tensor
        auto output = torch::zeros(grid_size, current_input.options());

        if (grid_size > 1) {
            // 情况 A: 数据量太大,需要多个 Block 合作
            // 这一步会将 n 个数据 压缩成 grid_size 个数据
            reduce_sum_kernel_v5<<<grid_size, block_size>>>(
                current_input.data_ptr<float>(),
                output.data_ptr<float>(),
                n
            );
        } else {
            // 情况 B: 数据量已经很小了 (<= 256),只需要 1 个 Block
            // 这一步直接算出最终结果
            reduce_sum_kernel_v1<<<1, block_size>>>( // 注意这里线程数要够覆盖 n
                current_input.data_ptr<float>(),
                output.data_ptr<float>(),
                n
            );
        }

        // 更新下一轮的输入
        current_input = output;
        n = grid_size;
    }

    return current_input;
}
"""


cpp_source = "torch::Tensor reduce_sum(torch::Tensor input);"

# 编译并加载模块
reduce_module = load_inline(
    name="reduce_extension",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["reduce_sum"],
    verbose=True,
)

# 测试代码
input_tensor = torch.ones(1024 * 1024 * 1024, device="cuda", dtype=torch.float32)
my_result = reduce_module.reduce_sum(input_tensor)
print(f"My Kernel Result: {my_result.item()}")
print(f"PyTorch Result: {torch.sum(input_tensor).item()}")


start_evt = torch.cuda.Event(enable_timing=True)
end_evt = torch.cuda.Event(enable_timing=True)

start_evt.record()
my_result = reduce_module.reduce_sum(input_tensor)
end_evt.record()
torch.cuda.synchronize() # 等待 GPU 跑完

print(f"Time: {start_evt.elapsed_time(end_evt):.3f} ms")

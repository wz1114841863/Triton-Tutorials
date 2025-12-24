import torch
from torch.utils.cpp_extension import load_inline

cuda_source = """
#include <cuda_runtime.h>
#include <stdio.h>

// A: MxK, B: KxN, C: MxN
// Grid(N/32, M/32),
// Block(32, 32)
// Naive 版本 (Baseline)
// 公式: 2 * M * N * K 次浮点运算
// 每一个 C 的元素 (M*N) 都要遍历一行 A 和一列 B (2K次读取).
// 总读取量: 2MNK
__global__ void sgemm_naive_kernel(float* A, float* B, float* C, int M, int N, int K) {
    // 1. 计算当前线程负责计算 C 的哪个元素 (row, col)
    // C[row, col] 对应 A 的 row 行和 B 的 col 列
    // 全局列 = 前面的 Block 占了多少列 + 我在当前 Block 的第几列
    int row = threadIdx.x + blockIdx.x * blockDim.x;
    // 全局行 = 前面的 Block 占了多少行 + 我在当前 Block 的第几行
    int col = threadIdx.y + blockIdx.y * blockDim.y;

    // 边界检查
    if (row < M && col < N) {
        float sum = 0.0f;

        // 2. 遍历 K 维度,计算点积
        // A 是行主序: A[row, k] = A[row * K + k]
        // B 是行主序: B[k, col] = B[k * N + col]
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }

        // 3. 写回结果
        C[row * N + col] = sum;
    }
}

// A: MxK, B: KxN, C: MxN
// Grid(N/32, M/32),
// Block(32, 32)
// 利用 Shared Memory 优化的版本
// Block 计算量: 一个 Block 负责 32 * 32 个元素.
// Phase (阶段) 数量: K / 32.
// 每个 Phase 的读取量: 32 * 32 (A) + 32 * 32 (B) = 2048
// 总读取量: 2048 * (K / 32) = 64 * K
// Grid 总读取量: (block数量) * (Block 读取量) = (M * N / 1024) * (64 * K) = 2(M * N * K) / 32
__global__ void sgemm_shared_mem_kernel(float* A, float* B, float* C, int M, int N, int K) {
    // Block 大小固定为 32x32
    const int BLOCK_SIZE = 32;

    // 1. 定义 Shared Memory
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    // 2. 定位线程在 Block 内的 ID (tx, ty) 和 全局 C 的坐标 (row, col)
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * BLOCK_SIZE + ty;
    int col = blockIdx.x * BLOCK_SIZE + tx;

    float sum = 0.0f;

    // 3. 主循环:沿着 K 维度,每次步进 BLOCK_SIZE
    // ph (phase) 表示当前是第几块
    for (int ph = 0; ph < K / BLOCK_SIZE; ++ph) {
        // --- 阶段 A: 协作加载数据到 Shared Memory ---
        // 加载 A 的元素到 As[ty][tx]
        // Global A 的行是 row,列是 (ph * BLOCK_SIZE + tx)
        As[ty][tx] = A[row * K + ph * BLOCK_SIZE + tx];

        // 加载 B 的元素到 Bs[ty][tx]
        // 提示: Global B 的行是 (ph * BLOCK_SIZE + ty),列是 col
        Bs[ty][tx] = B[(ph * BLOCK_SIZE + ty) * N + col];
        // 等待加载完成
        __syncthreads();

        // --- 阶段 B: 使用 Shared Memory 计算 ---
        // 遍历 Shared Memory 的 32 个元素进行累加
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }

        // 等待所有线程计算完, 才能进入下一轮覆盖 Shared Memory
        __syncthreads();
    }

    // 4. 写回结果
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

__global__ void sgemm_2d_tiling_kernel(float* A, float* B, float* C, int M, int N, int K) {
    // 配置参数
    const int BM = 32;
    const int BN = 32;
    const int BK = 8;
    const int TM = 4;
    const int TN = 4;

    // 1. 定义寄存器 (C的累加器)
    // 每个线程负责 4x4 的块
    float thread_results[TM][TN] = {0.0f};

    // 2. 定义 Shared Memory
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    // 3. 线程索引计算
    int tid = threadIdx.y * blockDim.x + threadIdx.x; // 0~63

    // 计算当前线程负责 C 的哪个子块 (row, col)
    // blockDim.x = 8, blockDim.y = 8
    // threadIdx.x (0~7) -> 对应 C 的列区域: threadIdx.x * TN
    // threadIdx.y (0~7) -> 对应 C 的行区域: threadIdx.y * TM
    int c_row = blockIdx.y * BM + threadIdx.y * TM;
    int c_col = blockIdx.x * BN + threadIdx.x * TN;

    // 寄存器缓存 A 和 B 的片段
    float reg_a[TM];
    float reg_b[TN];

    // --- 加载相关的预计算 ---
    // 64个线程搬运 As (32x8 = 256 float)
    // 每个线程搬 4 个 float
    // 我们可以把 As 看作一个 1D 数组 [256]
    // 线程 tid 负责搬运: tid*4, tid*4+1, tid*4+2, tid*4+3
    // 需要把这些 1D 索引映射回 (row, col) 以从 Global Memory 读取

    // 4. 主循环 (K 维度)
    for (int bk_idx = 0; bk_idx < K; bk_idx += BK) {

        // --- Phase A: 协作加载 ---
        // 如何把 64 个线程映射到 32x8 的数据上?
        // 简单方案: 把 As 当作 1D 线性内存.每个线程搬 4 个连续的数.
        // A 的 Global 指针: A + (blockIdx.y * BM) * K + bk_idx
        // :Stided Loop
        // 遍历 As 的所有元素,步长为 Total_Threads (64)
        for (int i = tid; i < BM * BK; i += 64) {
            int row = i / BK;
            int col = i % BK;
            As[row][col] = A[(blockIdx.y * BM + row) * K + (bk_idx + col)];
        }

        for (int i = tid; i < BK * BN; i += 64) {
            int row = i / BN;
            int col = i % BN;
            Bs[row][col] = B[(bk_idx + row) * N + (blockIdx.x * BN + col)];
        }

        __syncthreads();

        // --- Phase B: 计算 (Register Tiling) ---
        // 核心:外积更新
        for (int k = 0; k < BK; ++k) {
            // 1. 把 A 的一列 (TM个) 读到寄存器
            for (int i = 0; i < TM; ++i) {
                reg_a[i] = As[threadIdx.y * TM + i][k];
            }
            // 2. 把 B 的一行 (TN个) 读到寄存器
            for (int j = 0; j < TN; ++j) {
                reg_b[j] = Bs[k][threadIdx.x * TN + j];
            }

            // 3. 外积计算: C += A_col * B_row
            for (int i = 0; i < TM; ++i) {
                for (int j = 0; j < TN; ++j) {
                    thread_results[i][j] += reg_a[i] * reg_b[j];
                }
            }
        }
        __syncthreads();
    }

    // 5. 写回结果
    for (int i = 0; i < TM; ++i) {
        for (int j = 0; j < TN; ++j) {
            int global_row = c_row + i;
            int global_col = c_col + j;
            if (global_row < M && global_col < N) {
                C[global_row * N + global_col] = thread_results[i][j];
            }
        }
    }
}

__global__ void sgemm_2d_tiling_float4_kernel(float* A, float* B, float* C, int M, int N, int K) {
    const int BM = 32;
    const int BN = 32;
    const int BK = 8;
    const int TM = 4;
    const int TN = 4;

    // 线程数 = (32/4) * (32/4) = 64
    const int num_threads = 64;
    int tid = threadIdx.y * blockDim.x + threadIdx.x;

    // 1. 寄存器准备
    float thread_results[TM][TN] = {0.0f};
    float reg_a[TM];
    float reg_b[TN];

    // Shared Memory
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    // --- 准备加载索引 (Vectorized Loading) ---
    // 每个线程搬运一个 float4.
    // 计算这个 float4 对应 A/B 的哪里.

    // A 的加载: As[32][8] -> 视作 [32][2] 的 float4 数组
    // tid (0~63) -> 映射到 (row, col_div_4)
    int load_a_row = tid / 2;        // 0~31
    int load_a_col = (tid % 2) * 4;  // 0 或 4

    // B 的加载: Bs[8][32] -> 视作 [8][8] 的 float4 数组
    int load_b_row = tid / 8;        // 0~7
    int load_b_col = (tid % 8) * 4;  // 0, 4, ..., 28

    // C 的全局坐标
    int c_row = blockIdx.y * BM + threadIdx.y * TM;
    int c_col = blockIdx.x * BN + threadIdx.x * TN;

    // A, B 的全局指针基址 (行主序)
    // A_ptr 指向当前 Block 负责的 A 的行带
    // B_ptr 指向当前 Block 负责的 B 的列带
    const float* A_ptr = A + (blockIdx.y * BM) * K;
    const float* B_ptr = B + (blockIdx.x * BN);

    for (int bk_idx = 0; bk_idx < K; bk_idx += BK) {

        // --- Phase A: 向量化加载 ---

        // 加载 A 的 float4
        // Global Addr: A_ptr + (load_a_row * K) + (bk_idx + load_a_col)
        // Shared Addr: As[load_a_row][load_a_col]
        float4 tmp_a = reinterpret_cast<const float4*>(&A_ptr[load_a_row * K + bk_idx + load_a_col])[0];
        reinterpret_cast<float4*>(&As[load_a_row][load_a_col])[0] = tmp_a;

        // 加载 B 的 float4
        // Global Addr: B_ptr + (bk_idx + load_b_row) * N + load_b_col
        // Shared Addr: Bs[load_b_row][load_b_col]
        float4 tmp_b = reinterpret_cast<const float4*>(&B_ptr[(bk_idx + load_b_row) * N + load_b_col])[0];
        reinterpret_cast<float4*>(&Bs[load_b_row][load_b_col])[0] = tmp_b;

        __syncthreads();

        // --- Phase B: 计算 (与之前逻辑一致) ---
        for (int k = 0; k < BK; ++k) {
            // 加载 A 的一列 (TM=4) 到寄存器
            // 这里不能向量化,因为是跨行的 (Stride Access)
            for (int i = 0; i < TM; ++i) {
                reg_a[i] = As[threadIdx.y * TM + i][k];
            }
            // 加载 B 的一行 (TN=4) 到寄存器
            // 这里可以向量化加载,但为了代码简单,先手动写
            for (int j = 0; j < TN; ++j) {
                reg_b[j] = Bs[k][threadIdx.x * TN + j];
            }

            // 外积计算
            for (int i = 0; i < TM; ++i) {
                for (int j = 0; j < TN; ++j) {
                    thread_results[i][j] += reg_a[i] * reg_b[j];
                }
            }
        }
        __syncthreads();
    }

    // 写回结果 (每个线程负责 TM * TN = 16 个点)
    for (int i = 0; i < TM; ++i) {
        for (int j = 0; j < TN; ++j) {
             if (c_row + i < M && c_col + j < N) {
                 C[(c_row + i) * N + (c_col + j)] = thread_results[i][j];
             }
        }
    }
}

__global__ void sgemm_big_tiling_float4_kernel(float* A, float* B, float* C, int M, int N, int K) {
    // 1. 增大分块配置
    const int BM = 64;
    const int BN = 64;
    const int BK = 8;
    const int TM = 8;
    const int TN = 8;

    const int tid = threadIdx.y * blockDim.x + threadIdx.x; // 0~63

    // 2. 寄存器 (每个线程算 8x8 = 64 个元素)
    // 这种大量寄存器使用在 RTX 3060 上是完全没问题的
    float thread_results[TM][TN] = {0.0f};
    float reg_a[TM]; // 8
    float reg_b[TN]; // 8

    // 3. Shared Memory
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    // 4. 准备加载索引 (Vectorized Loading)
    // ---------------------------------------------------------
    // 我们的目标:64个线程协作搬运 As (64x8 = 512 floats = 128 float4)
    // 每个线程需要搬运: 128 / 64 = 2 个 float4
    // 策略:Thread i 搬运第 i 个和第 (i+64) 个 float4
    // ---------------------------------------------------------

    // A 的搬运信息 (视作 [64][2] 的 float4 数组)
    // 每一个 float4 包含 4 个 float, 对应 A 的半行
    int load_a_row_1 = tid / 2;        // 0~31
    int load_a_col_1 = (tid % 2) * 4;
    int load_a_row_2 = load_a_row_1 + 32; // 32~63 (处理下半部分)
    int load_a_col_2 = load_a_col_1;

    // B 的搬运信息 (视作 [8][16] 的 float4 数组)
    // 总共 8行 * 64列 = 512 floats = 128 float4
    // 同样,Thread i 搬运第 i 个和第 (i+64) 个 float4
    // row = idx / 16, col = (idx % 16) * 4
    int load_b_row_1 = tid / 16;       // 0~3
    int load_b_col_1 = (tid % 16) * 4;
    int load_b_row_2 = load_b_row_1 + 4; // 4~7 (处理下半部分)
    int load_b_col_2 = load_b_col_1;

    // A, B 的全局指针基址
    const float* A_ptr = A + (blockIdx.y * BM) * K;
    const float* B_ptr = B + (blockIdx.x * BN);

    // C 的全局坐标
    int c_row = blockIdx.y * BM + threadIdx.y * TM;
    int c_col = blockIdx.x * BN + threadIdx.x * TN;

    for (int bk_idx = 0; bk_idx < K; bk_idx += BK) {

        // --- Phase A: 向量化加载 (每人搬 2 次) ---

        // Load A (Part 1)
        float4 tmp_a1 = reinterpret_cast<const float4*>(&A_ptr[load_a_row_1 * K + bk_idx + load_a_col_1])[0];
        reinterpret_cast<float4*>(&As[load_a_row_1][load_a_col_1])[0] = tmp_a1;

        // Load A (Part 2)
        float4 tmp_a2 = reinterpret_cast<const float4*>(&A_ptr[load_a_row_2 * K + bk_idx + load_a_col_2])[0];
        reinterpret_cast<float4*>(&As[load_a_row_2][load_a_col_2])[0] = tmp_a2;

        // Load B (Part 1)
        float4 tmp_b1 = reinterpret_cast<const float4*>(&B_ptr[(bk_idx + load_b_row_1) * N + load_b_col_1])[0];
        reinterpret_cast<float4*>(&Bs[load_b_row_1][load_b_col_1])[0] = tmp_b1;

        // Load B (Part 2)
        float4 tmp_b2 = reinterpret_cast<const float4*>(&B_ptr[(bk_idx + load_b_row_2) * N + load_b_col_2])[0];
        reinterpret_cast<float4*>(&Bs[load_b_row_2][load_b_col_2])[0] = tmp_b2;

        __syncthreads();

        // --- Phase B: 计算 (8x8 外积) ---
        // 修改这里的循环上限为 TM, TN, BK
        for (int k = 0; k < BK; ++k) {
            // 加载 A 的一列 (TM=8) 到寄存器
            for (int i = 0; i < TM; ++i) {
                reg_a[i] = As[threadIdx.y * TM + i][k];
            }
            // 加载 B 的一行 (TN=8) 到寄存器
            for (int j = 0; j < TN; ++j) {
                reg_b[j] = Bs[k][threadIdx.x * TN + j];
            }

            // 外积计算 8x8
            for (int i = 0; i < TM; ++i) {
                for (int j = 0; j < TN; ++j) {
                    thread_results[i][j] += reg_a[i] * reg_b[j];
                }
            }
        }
        __syncthreads();
    }

    // 5. 写回结果 (8x8)
    for (int i = 0; i < TM; ++i) {
        for (int j = 0; j < TN; ++j) {
             if (c_row + i < M && c_col + j < N) {
                 C[(c_row + i) * N + (c_col + j)] = thread_results[i][j];
             }
        }
    }
}

// -------------------------------------------------------------------------
// Kernel: 128x128 Block Tile, 8x8 Thread Tile, Vectorized Load (float4)
// -------------------------------------------------------------------------
__global__ void sgemm_128x128_k8_kernel(float* A, float* B, float* C, int M, int N, int K) {
    // 1. 配置参数
    const int BM = 128;
    const int BN = 128;
    const int BK = 8;
    const int TM = 8;
    const int TN = 8;

    // 当前线程在 Block 内的索引 (0 ~ 255)
    int tid = threadIdx.y * blockDim.x + threadIdx.x;

    // 2. 寄存器准备 (每个线程算 8x8 = 64 个元素)
    float thread_results[TM][TN] = {0.0f};
    float reg_a[TM];
    float reg_b[TN];

    // 3. Shared Memory
    // A: 128行 * 8列; B: 8行 * 128列
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    // 4. 准备加载索引
    // -------------------------------------------------------------------
    // 线程数 = 256
    // A 需要搬运 128x8 = 1024 floats = 256 float4 -> 每人 1 个 float4
    // B 需要搬运 8x128 = 1024 floats = 256 float4 -> 每人 1 个 float4
    // -------------------------------------------------------------------

    // A 的加载索引: As[128][8] 视作 float4 数组 [128][2]
    // tid (0~255) -> 映射到行 (0~127) 和 列 (0或4)
    // 比如: tid=0 -> row=0, col=0; tid=1 -> row=0, col=4; tid=2 -> row=1, col=0
    int load_a_row = tid / 2;
    int load_a_col = (tid % 2) * 4;

    // B 的加载索引: Bs[8][128] 视作 float4 数组 [8][32]
    // tid (0~255) -> 映射到行 (0~7) 和 列 (0, 4, ... 124)
    int load_b_row = tid / 32;
    int load_b_col = (tid % 32) * 4;

    // A, B 的全局指针基址
    const float* A_ptr = A + (blockIdx.y * BM) * K;
    const float* B_ptr = B + (blockIdx.x * BN);

    // C 的全局坐标
    int c_row = blockIdx.y * BM + threadIdx.y * TM;
    int c_col = blockIdx.x * BN + threadIdx.x * TN;

    // --- 主循环 ---
    for (int bk_idx = 0; bk_idx < K; bk_idx += BK) {

        // --- Phase A: 向量化加载 (每人只搬 1 次!) ---
        // Load A (Global -> Shared)
        float4 tmp_a = reinterpret_cast<const float4*>(&A_ptr[load_a_row * K + bk_idx + load_a_col])[0];
        reinterpret_cast<float4*>(&As[load_a_row][load_a_col])[0] = tmp_a;

        // Load B (Global -> Shared)
        float4 tmp_b = reinterpret_cast<const float4*>(&B_ptr[(bk_idx + load_b_row) * N + load_b_col])[0];
        reinterpret_cast<float4*>(&Bs[load_b_row][load_b_col])[0] = tmp_b;

        __syncthreads();

        // --- Phase B: 计算 (8x8 外积) ---
        for (int k = 0; k < BK; ++k) {
            // 取出 A 的一列 (8个)
            for (int i = 0; i < TM; ++i) {
                reg_a[i] = As[threadIdx.y * TM + i][k];
            }
            // 取出 B 的一行 (8个)
            for (int j = 0; j < TN; ++j) {
                reg_b[j] = Bs[k][threadIdx.x * TN + j];
            }
            // FMA
            for (int i = 0; i < TM; ++i) {
                for (int j = 0; j < TN; ++j) {
                    thread_results[i][j] += reg_a[i] * reg_b[j];
                }
            }
        }
        __syncthreads();
    }

    // 5. 写回结果 (8x8)
    for (int i = 0; i < TM; ++i) {
        for (int j = 0; j < TN; ++j) {
             if (c_row + i < M && c_col + j < N) {
                 C[(c_row + i) * N + (c_col + j)] = thread_results[i][j];
             }
        }
    }
}

// -------------------------------------------------------------------------
// Kernel: BCF (Bank Conflict Free) + Transpose A + Padding
// -------------------------------------------------------------------------
__global__ void sgemm_128x128_k8_bcf_kernel(float* A, float* B, float* C, int M, int N, int K) {
    const int BM = 128;
    const int BN = 128;
    const int BK = 8;
    const int TM = 8;
    const int TN = 8;
    const int OFFSET = 4; // Padding 大小,保证 float4 对齐

    int tid = threadIdx.y * blockDim.x + threadIdx.x;

    // 1. Shared Memory (注意维度变化和 Padding)
    // s_a 转置为 [BK][BM],方便计算时 float4 读取
    __shared__ float s_a[BK][BM + OFFSET];
    // s_b 保持 [BK][BN],加 Padding 减少读取冲突
    __shared__ float s_b[BK][BN + OFFSET];

    // 寄存器
    float thread_results[TM][TN] = {0.0f};
    float reg_a[TM];
    float reg_b[TN];

    // 2. 加载索引计算
    // -------------------------------------------------------------------
    // A (Global): 128行 x 8列.每人搬 1 个 float4 (包含了 A[row][k]...A[row][k+3])
    // 我们需要把这个 float4 竖着写进 s_a (因为 s_a 转置了)
    // -------------------------------------------------------------------
    int load_a_row = tid / 2;       // 0~127
    int load_a_col = (tid % 2) * 4; // 0 或 4

    // B (Global): 8行 x 128列.
    int load_b_row = tid / 32;       // 0~7
    int load_b_col = (tid % 32) * 4; // 0, 4, ... 124

    const float* A_ptr = A + (blockIdx.y * BM) * K;
    const float* B_ptr = B + (blockIdx.x * BN);

    int c_row = blockIdx.y * BM + threadIdx.y * TM;
    int c_col = blockIdx.x * BN + threadIdx.x * TN;

    for (int bk_idx = 0; bk_idx < K; bk_idx += BK) {

        // --- Phase A: 加载 (含 Transpose A) ---

        // 1. 加载 A 的 float4 到寄存器
        float4 tmp_a = reinterpret_cast<const float4*>(&A_ptr[load_a_row * K + bk_idx + load_a_col])[0];

        // 2. 转置写入 s_a
        // Global A 是 A[row][k], A[row][k+1]...
        // tmp_a.x = A[row][k]   -> 写入 s_a[0][row] (假设 load_a_col=0)
        // tmp_a.y = A[row][k+1] -> 写入 s_a[1][row]
        // 注意:load_a_col 可能是 0 或 4,对应 s_a 的行索引
        s_a[load_a_col + 0][load_a_row] = tmp_a.x;
        s_a[load_a_col + 1][load_a_row] = tmp_a.y;
        s_a[load_a_col + 2][load_a_row] = tmp_a.z;
        s_a[load_a_col + 3][load_a_row] = tmp_a.w;

        // 3. 加载 B (直接写入,无需转置)
        float4 tmp_b = reinterpret_cast<const float4*>(&B_ptr[(bk_idx + load_b_row) * N + load_b_col])[0];
        // 使用 FLOAT4 宏强转写入,利用 Padding
        reinterpret_cast<float4*>(&s_b[load_b_row][load_b_col])[0] = tmp_b;

        __syncthreads();

        // --- Phase B: 计算 (全 float4 读取) ---
        for (int k = 0; k < BK; ++k) {
            // 4. 读取 reg_a (从 s_a)
            // 现在的 s_a 是 [BK][BM],所以 s_a[k][...] 是连续的!
            // 我们可以一次读 4 个 float!
            // 需要读 TM=8 个,所以读 2 次 float4

            // 下面这几行是性能提升的关键:向量化读取 Shared Memory
            float4 ra1 = reinterpret_cast<float4*>(&s_a[k][threadIdx.y * TM])[0];     // load a[0..3]
            float4 ra2 = reinterpret_cast<float4*>(&s_a[k][threadIdx.y * TM + 4])[0]; // load a[4..7]

            reg_a[0] = ra1.x; reg_a[1] = ra1.y; reg_a[2] = ra1.z; reg_a[3] = ra1.w;
            reg_a[4] = ra2.x; reg_a[5] = ra2.y; reg_a[6] = ra2.z; reg_a[7] = ra2.w;

            // 5. 读取 reg_b (从 s_b)
            // s_b 是 [BK][BN],s_b[k][...] 也是连续的
            float4 rb1 = reinterpret_cast<float4*>(&s_b[k][threadIdx.x * TN])[0];
            float4 rb2 = reinterpret_cast<float4*>(&s_b[k][threadIdx.x * TN + 4])[0];

            reg_b[0] = rb1.x; reg_b[1] = rb1.y; reg_b[2] = rb1.z; reg_b[3] = rb1.w;
            reg_b[4] = rb2.x; reg_b[5] = rb2.y; reg_b[6] = rb2.z; reg_b[7] = rb2.w;

            // FMA
            for (int i = 0; i < TM; ++i) {
                for (int j = 0; j < TN; ++j) {
                    thread_results[i][j] += reg_a[i] * reg_b[j];
                }
            }
        }
        __syncthreads();
    }

    // 写回结果 (保持不变)
    for (int i = 0; i < TM; ++i) {
        for (int j = 0; j < TN; ++j) {
             if (c_row + i < M && c_col + j < N) {
                 C[(c_row + i) * N + (c_col + j)] = thread_results[i][j];
             }
        }
    }
}

__global__ void sgemm_128x128_k8_dbuf_kernel(float* A, float* B, float* C, int M, int N, int K) {
    // 配置参数
    const int BM = 128;
    const int BN = 128;
    const int BK = 8;
    const int TM = 8;
    const int TN = 8;
    const int OFFSET = 4; // Padding 大小,保证 float4 对齐

    // Block 内线程 ID
    int tid = threadIdx.y * blockDim.x + threadIdx.x;

    // 1. 双缓冲 Shared Memory
    __shared__ float s_a[2][BK][BM + OFFSET];
    __shared__ float s_b[2][BK][BN + OFFSET];

    float thread_results[TM][TN] = {0.0f};
    float reg_a[TM];
    float reg_b[TN];

    // 2. 增加寄存器用于Prefetch
    // 暂存从GLobal Memory 读来的下一块数据
    float4 load_reg_a;
    float4 load_reg_b;

    // 3. 计算加载索引
    int load_a_row = tid / 2;       // 0~127
    int load_a_col = (tid % 2) * 4; // 0 或 4
    int load_b_row = tid / 32;       // 0~7
    int load_b_col = (tid % 32) * 4; // 0, 4, ... 124
    // 第 blockIdx.y 个 Block 负责的 A 行带
    const float* A_ptr = A + (blockIdx.y * BM) * K;
    // 第 blockIdx.x 个 Block 负责的 B 列带
    const float* B_ptr = B + (blockIdx.x * BN);

    int c_row = blockIdx.y * BM + threadIdx.y * TM;
    int c_col = blockIdx.x * BN + threadIdx.x * TN;

    // Prolog (序幕): 加载第 0 块数据到 Shared Memory Buffer 0
    {
        // Global -> Register
        load_reg_a = reinterpret_cast<const float4*>(&A_ptr[load_a_row * K + 0 + load_a_col])[0];
        load_reg_b = reinterpret_cast<const float4*>(&B_ptr[(0 + load_b_row) * N + load_b_col])[0];

        // Register -> Shared Memory (Buffer 0)
        // Transpose A
        s_a[0][load_a_col + 0][load_a_row] = load_reg_a.x;
        s_a[0][load_a_col + 1][load_a_row] = load_reg_a.y;
        s_a[0][load_a_col + 2][load_a_row] = load_reg_a.z;
        s_a[0][load_a_col + 3][load_a_row] = load_reg_a.w;

        // load B
        reinterpret_cast<float4*>(&s_b[0][load_b_row][load_b_col])[0] = load_reg_b;
    }
    __syncthreads();

    // Main Loop (流水线): 从第 1 块开始循环
    int write_stage_idx = 1; // 准备写入 Buffer 1
    int read_stage_idx = 0;  // 准备读取 Buffer 0

    // 注意:bk_idx 从 BK 开始,而不是 0
    for (int bk_idx = BK; bk_idx < K; bk_idx += BK) {
        // 1. load next block from Global to Register
        // 此时计算单元正在用Shared Memory, 所以可以并行加载下一块数据
        load_reg_a = reinterpret_cast<const float4*>(&A_ptr[load_a_row * K + bk_idx + load_a_col])[0];
        load_reg_b = reinterpret_cast<const float4*>(&B_ptr[(bk_idx + load_b_row) * N + load_b_col])[0];

        // 2. compute current
        // 计算和加载是并行的
        for (int k = 0; k < BK; ++k) {
            // 从s_a[read]读取
            float4 ra1 = reinterpret_cast<float4*>(&s_a[read_stage_idx][k][threadIdx.y * TM])[0];
            float4 ra2 = reinterpret_cast<float4*>(&s_a[read_stage_idx][k][threadIdx.y * TM + 4])[0];

            // 从s_b[read]读取
            float4 rb1 = reinterpret_cast<float4*>(&s_b[read_stage_idx][k][threadIdx.x * TN])[0];
            float4 rb2 = reinterpret_cast<float4*>(&s_b[read_stage_idx][k][threadIdx.x * TN + 4])[0];

            // 展开到寄存器
            reg_a[0] = ra1.x; reg_a[1] = ra1.y; reg_a[2] = ra1.z; reg_a[3] = ra1.w;
            reg_a[4] = ra2.x; reg_a[5] = ra2.y; reg_a[6] = ra2.z; reg_a[7] = ra2.w;
            reg_b[0] = rb1.x; reg_b[1] = rb1.y; reg_b[2] = rb1.z; reg_b[3] = rb1.w;
            reg_b[4] = rb2.x; reg_b[5] = rb2.y; reg_b[6] = rb2.z; reg_b[7] = rb2.w;

            for (int i = 0; i < TM; ++i) {
                for (int j = 0; j < TN; ++j) {
                    thread_results[i][j] += reg_a[i] * reg_b[j];
                }
            }
        }

        // 3. Store next block from Register to Shared Memory (write stage)
        // 写入到 [write] Buffer(另一个), 所以完全安全,不用同步
        // Transpose A to write buffer
        s_a[write_stage_idx][load_a_col + 0][load_a_row] = load_reg_a.x;
        s_a[write_stage_idx][load_a_col + 1][load_a_row] = load_reg_a.y;
        s_a[write_stage_idx][load_a_col + 2][load_a_row] = load_reg_a.z;
        s_a[write_stage_idx][load_a_col + 3][load_a_row] = load_reg_a.w;
        // Store B to write buffer
        reinterpret_cast<float4*>(&s_b[write_stage_idx][load_b_row][load_b_col])[0] = load_reg_b;

        // 4. syncthreads
        __syncthreads();

        // 交换 Ping-Pong 指针
        // read 变 write, write 变 read
        // 使用异或快速切换 0/1
        write_stage_idx ^= 1;
        read_stage_idx ^= 1;
    }

    // Epilog (收尾): 计算最后一块数据
    // 循环结束后,最后一块数据已经加载到 read_stage_idx 指向的 Buffer 了
    // 但还没算,所以这里补上最后一次计算
    for (int k = 0; k < BK; ++k) {
        float4 ra1 = reinterpret_cast<float4*>(&s_a[read_stage_idx][k][threadIdx.y * TM])[0];
        float4 ra2 = reinterpret_cast<float4*>(&s_a[read_stage_idx][k][threadIdx.y * TM + 4])[0];
        float4 rb1 = reinterpret_cast<float4*>(&s_b[read_stage_idx][k][threadIdx.x * TN])[0];
        float4 rb2 = reinterpret_cast<float4*>(&s_b[read_stage_idx][k][threadIdx.x * TN + 4])[0];

        reg_a[0] = ra1.x; reg_a[1] = ra1.y; reg_a[2] = ra1.z; reg_a[3] = ra1.w;
        reg_a[4] = ra2.x; reg_a[5] = ra2.y; reg_a[6] = ra2.z; reg_a[7] = ra2.w;
        reg_b[0] = rb1.x; reg_b[1] = rb1.y; reg_b[2] = rb1.z; reg_b[3] = rb1.w;
        reg_b[4] = rb2.x; reg_b[5] = rb2.y; reg_b[6] = rb2.z; reg_b[7] = rb2.w;

        for (int i = 0; i < TM; ++i) {
            for (int j = 0; j < TN; ++j) {
                thread_results[i][j] += reg_a[i] * reg_b[j];
            }
        }
    }

    // 写回结果 (保持不变)
    for (int i = 0; i < TM; ++i) {
        for (int j = 0; j < TN; ++j) {
             if (c_row + i < M && c_col + j < N) {
                 C[(c_row + i) * N + (c_col + j)] = thread_results[i][j];
             }
        }
    }
}
torch::Tensor launch_sgemm_naive(torch::Tensor A, torch::Tensor B) {
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    auto C = torch::empty({M, N}, torch::dtype(torch::kFloat32).device(A.device()));

    // dim3 block(32, 32);
    dim3 block(8, 8);

    // dim3 grid((N + 32 - 1) / 32, (M + 32 - 1) / 32);
    dim3 grid((N + 64 - 1) / 64, (M + 64 - 1) / 64);

    sgemm_big_tiling_float4_kernel<<<grid, block>>>(
        (float*)A.data_ptr(),
        (float*)B.data_ptr(),
        (float*)C.data_ptr(),
        M, N, K
    );
    return C;
}

torch::Tensor launch_sgemm_128x128(torch::Tensor A, torch::Tensor B) {
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    auto C = torch::empty({M, N}, torch::dtype(torch::kFloat32).device(A.device()));

    // 配置 Block (16, 16)
    dim3 block(16, 16);
    // 配置 Grid (按 128 切分)
    dim3 grid((N + 128 - 1) / 128, (M + 128 - 1) / 128);

    sgemm_128x128_k8_dbuf_kernel<<<grid, block>>>(
        (float*)A.data_ptr(),
        (float*)B.data_ptr(),
        (float*)C.data_ptr(),
        M, N, K
    );
    return C;
}

"""

cpp_source = "torch::Tensor launch_sgemm_128x128(torch::Tensor A, torch::Tensor B);"

module = load_inline(
    name="sgemm_128x128",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["launch_sgemm_128x128"],
    extra_cuda_cflags=["-O3"],
)

# --- 测试 ---
M, N, K = 1024, 1024, 1024
A = torch.randn(M, K, device="cuda")
B = torch.randn(K, N, device="cuda")

# 运行自定义算子
C_my = module.launch_sgemm_128x128(A, B)
print("My C shape:", C_my.shape)
# 运行 PyTorch (cuBLAS)
C_torch = torch.matmul(A, B)
print("Torch C shape:", C_torch.shape)

# 允许一定的浮点误差 (因为累加顺序不同)
if torch.allclose(C_my, C_torch, atol=1e-3):
    print("✅ GEMM Match!")
else:
    print("❌ Mismatch!")
    print("Diff:", (C_my - C_torch).abs().max().item())

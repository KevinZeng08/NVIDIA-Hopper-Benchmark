#include <cuda.h>
#include <cuda/barrier>
#include <cuda_fp16.h>          // half
#include <cuda_bf16.h>          // __nv_bfloat16
#include "../../util.h"

using barrier = cuda::barrier<cuda::thread_scope_block>;

// Data type configuration
// Uncomment ONE of the following lines to select data type:
// #define USE_FLOAT32
// #define USE_FLOAT16
#define USE_BFLOAT16

#if defined(USE_FLOAT16)
    typedef half dtype;
    #define DTYPE_NAME "FP16"
#elif defined(USE_BFLOAT16)
    typedef __nv_bfloat16 dtype;
    #define DTYPE_NAME "BF16"
#else  // USE_FLOAT32 or default
    typedef float dtype;
    #define DTYPE_NAME "FP32"
#endif

#define ARRAY_SIZE (4 * 1024*1024*(1024/sizeof(dtype))) // GB
constexpr uint SMEM_WIDTH[] = {64, 64, 64, 64, 64, 64, 64, 128, 128, 256, 256}; // 1-128 KB
constexpr uint  SMEM_HEIGHT[] = {1, 2, 4, 8, 16, 32, 64, 64, 128, 128, 256}; // sizeof(float) * 2*32 * 2*32 equals to LOAD_SIZE 
constexpr uint BLOCKS[] = {132}; // same as number of SMs
#define THREADS_PER_BLOCK 128
constexpr uint IDX = 1;
constexpr uint LOAD_SIZE = (SMEM_WIDTH[IDX] * SMEM_HEIGHT[IDX] * sizeof(dtype)); //bytes
constexpr int CP_ASYNC_BYTES = 16; // 16-byte each thread for cp.async.cg

// cp.async helper functions
__device__ inline void cp_async_16(void *smem_dst, const void *global_src) {
  unsigned long long dst_s, src_g;
  asm volatile("cvta.to.shared.u64 %0, %1;" : "=l"(dst_s) : "l"(smem_dst));
  asm volatile("cvta.to.global.u64 %0, %1;" : "=l"(src_g) : "l"(global_src));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"l"(dst_s), "l"(src_g));
}

template <int group_size = 0>
__device__ inline void cp_async_wait_group() {
  asm volatile("cp.async.wait_group %0;" ::"n"(group_size));
}

__device__ inline void cp_async_commit_group() {
  asm volatile("cp.async.commit_group;");
}

__global__ void init_data(dtype * array) {
    uint32_t tid = threadIdx.x;
    uint32_t uid = blockIdx.x * blockDim.x + tid;
    auto total_threads = blockDim.x * gridDim.x;

    for (uint32_t i = uid; i < ARRAY_SIZE; i += total_threads) {
#if defined(USE_FLOAT16)
        array[i] = __float2half((float)uid);
#elif defined(USE_BFLOAT16)
        array[i] = __float2bfloat16((float)uid);
#else
        array[i] = uid;
#endif
    }
}

// Multi-thread CP.ASYNC 2D bandwidth kernel (linear offset version)
__global__ void cp_async_bw_2d(dtype *array, dtype *dsink)
{
    uint32_t tid = threadIdx.x;
    uint32_t block_offset = blockIdx.x;
    // dtype temp_res = 0;

    // __shared__ alignas(16) dtype smem[LOAD_SIZE/sizeof(dtype)];
    extern __shared__ __align__(128) dtype smem[];
#pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ barrier bar;
    if (tid == 0) {
        init(&bar, blockDim.x);
        asm volatile("fence.proxy.async.shared::cta;");     // b)
    }
    __syncthreads();

    const int total_tiles = ARRAY_SIZE * sizeof(dtype) / LOAD_SIZE;

    // Each block processes tiles with stride
    for (int tile_idx = block_offset; tile_idx < total_tiles; tile_idx += gridDim.x) {

        // Linear offset calculation (no 2D coordinate computation)
        size_t base_offset = tile_idx * (LOAD_SIZE / sizeof(dtype));

        // All threads cooperatively load the tile using cp.async
        for (int byte_offset = tid * CP_ASYNC_BYTES;
             byte_offset < LOAD_SIZE;
             byte_offset += blockDim.x * CP_ASYNC_BYTES) {

            dtype *src_ptr = array + base_offset + byte_offset / sizeof(dtype);
            dtype *dst_ptr = smem + byte_offset / sizeof(dtype);
            cp_async_16(dst_ptr, src_ptr);
        }

        // Commit and wait
        cp_async_commit_group();
        cp_async_wait_group<0>();
        // temp_res += smem[0];
        // 3b. All threads arrive on the barrier
        barrier::arrival_token token = bar.arrive();

        // 3c. Wait for the data to have arrived.
        bar.wait(std::move(token));
    }
}


int main() {

    for (int i = 0; i < sizeof(BLOCKS)/sizeof(int); ++i) {
        printf("\n=== CP.ASYNC 2D Benchmark [%s] ===\n", DTYPE_NAME);
        printf("Data type size: %zu bytes\n", sizeof(dtype));
        printf("Block size = %d, Threads = %d\n", BLOCKS[i], THREADS_PER_BLOCK);
        printf("Tile dimensions: Width = %d, Height = %d\n", SMEM_WIDTH[IDX], SMEM_HEIGHT[IDX]);
        printf("Load size per tile = %.2f KB\n", (float)LOAD_SIZE/1024);

        dtype *dsink = (dtype *)malloc(sizeof(dtype));
        dtype *array_g;
        dtype *dsink_g;

        CUDA_CHECK(cudaMalloc(&array_g, sizeof(dtype) * ARRAY_SIZE));
        CUDA_CHECK(cudaMalloc(&dsink_g, sizeof(dtype)));

        init_data<<<BLOCKS[i], THREADS_PER_BLOCK>>>(array_g);

        auto kernel = &cp_async_bw_2d;
        int smem_size = LOAD_SIZE;
        if (smem_size >= 48 * 1024) {
            cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
        }
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        float milliseconds = 0;

        // Test 2D pattern version
        cudaEventRecord(start);
        kernel<<<BLOCKS[i], THREADS_PER_BLOCK, smem_size>>>(array_g, dsink_g);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        CUDA_CHECK(cudaPeekAtLastError());

        cudaEventElapsedTime(&milliseconds, start, stop);

        printf("\n[CP.ASYNC version]\n");
        printf("Total time = %f ms, transfer size = %lu bytes\n", milliseconds, ARRAY_SIZE * sizeof(dtype));
        printf("Throughput: %.2f GB/s\n\n", ARRAY_SIZE * sizeof(dtype) / (milliseconds / 1000) / 1024 / 1024 / 1024);

        cudaFree(array_g);
        cudaFree(dsink_g);
        free(dsink);
    }

    return 0;
}

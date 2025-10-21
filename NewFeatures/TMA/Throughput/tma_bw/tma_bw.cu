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
// constexpr int BLOCKS[] = {114, 228, 342, 456};
constexpr int BLOCKS[] = {132};
#define THREADS_PER_BLOCK 128

constexpr int LOAD_SIZE_LIST[] = {
    // 64*4*sizeof(dtype), 64*8*sizeof(dtype), // cannot be used for multi-thread TMA because each thread requires at least loading 16 bytes
                                    64*16*sizeof(dtype), 64*32*sizeof(dtype), 64*64*sizeof(dtype), 
                                    128*64*sizeof(dtype), 128*128*sizeof(dtype),
                                    128*256*sizeof(dtype), 128*512*sizeof(dtype)}; //bytes 4KB-128KB
constexpr int LOAD_SIZE = LOAD_SIZE_LIST[1];
// Auto-calculate TMA bytes per thread based on LOAD_SIZE and THREADS_PER_BLOCK
constexpr int TMA_BYTES_PER_THREAD = LOAD_SIZE / THREADS_PER_BLOCK;

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

__global__ void tma_bw(dtype * volatile array, dtype *dsink)
{

    uint32_t tid = threadIdx.x;
	// uint32_t uid = blockIdx.x * blockDim.x + tid;
    uint32_t block_offset = blockIdx.x;
    // dtype temp_res = 0;

    // __shared__ alignas(16) dtype smem[LOAD_SIZE/sizeof(dtype)];
    extern __shared__ __align__(128) dtype smem[];

#pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ barrier bar;
    if (tid == 0) {
        init(&bar, 1);                    // a) only one thread issues TMA copy, so only wait for 1 thread
        asm volatile("fence.proxy.async.shared::cta;");     // b)

        // for (int i = uid * (LOAD_SIZE / sizeof(dtype)); i < ARRAY_SIZE; i += gridDim.x * blockDim.x * (LOAD_SIZE / sizeof(dtype))) {
        for (int i = block_offset; i < ARRAY_SIZE / (LOAD_SIZE / sizeof(dtype)); i += gridDim.x * 1) {

            auto ptr = array + i * (LOAD_SIZE / sizeof(dtype));

            asm volatile(
                "{\t\n"
                //"discard.L2 [%1], 128;\n\t"
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes[%0], [%1], %2, [%3]; // 1a. unicast\n\t"
                "mbarrier.expect_tx.relaxed.cta.shared::cta.b64 [%3], %2;\n\t"
                "}"
                :
                //: "r"(static_cast<unsigned>(__cvta_generic_to_shared(ptr))), "l"(ptr[0]), "n"(cuda::aligned_size_t<16>(LOAD_SIZE)), "r"(static_cast<unsigned>(__cvta_generic_to_shared(&bar)))
                : "r"(static_cast<unsigned>(__cvta_generic_to_shared(smem))), "l"(ptr), "n"(LOAD_SIZE), "r"(static_cast<unsigned>(__cvta_generic_to_shared(&bar)))
                : "memory"); 


            // 3b. All threads arrive on the barrier
            barrier::arrival_token token = bar.arrive();

            // 3c. Wait for the data to have arrived.
            bar.wait(std::move(token));
            //temp_res += smem[0];
        }


    }


}

__global__ void tma_bw_multi_thread(dtype * volatile array, dtype *dsink)
{

    uint32_t tid = threadIdx.x;
	// uint32_t uid = blockIdx.x * blockDim.x + tid;
    uint32_t block_offset = blockIdx.x;
    // dtype temp_res = 0;

    // __shared__ alignas(16) dtype smem[LOAD_SIZE/sizeof(dtype)];
    extern __shared__ __align__(128) dtype smem[];

#pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ barrier bar;
    if (tid == 0) {
        init(&bar, blockDim.x);  // All threads participate in barrier
        asm volatile("fence.proxy.async.shared::cta;");     // b)
    }
    __syncthreads();

    const int total_tiles = ARRAY_SIZE * sizeof(dtype) / LOAD_SIZE;

    // Each block processes tiles with stride
    for (int tile_idx = block_offset; tile_idx < total_tiles; tile_idx += gridDim.x) {

        // Linear offset calculation
        size_t base_offset = tile_idx * (LOAD_SIZE / sizeof(dtype));

        // All threads cooperatively load the tile using TMA
        // Each thread loads TMA_BYTES_PER_THREAD bytes (128x1 elements)
        for (int byte_offset = tid * TMA_BYTES_PER_THREAD;
             byte_offset < LOAD_SIZE;
             byte_offset += blockDim.x * TMA_BYTES_PER_THREAD) {

            auto src_ptr = array + base_offset + byte_offset / sizeof(dtype);
            auto dst_ptr = smem + byte_offset / sizeof(dtype);

            asm volatile(
                "{\t\n"
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes[%0], [%1], %2, [%3];\n\t"
                "mbarrier.expect_tx.relaxed.cta.shared::cta.b64 [%3], %2;\n\t"
                "}"
                :
                : "r"(static_cast<unsigned>(__cvta_generic_to_shared(dst_ptr))),
                  "l"(src_ptr),
                  "n"(TMA_BYTES_PER_THREAD),
                  "r"(static_cast<unsigned>(__cvta_generic_to_shared(&bar)))
                : "memory");
        }

        // All threads arrive on the barrier
        barrier::arrival_token token = bar.arrive();

        // Wait for the data to have arrived
        bar.wait(std::move(token));
        //temp_res += smem[0];
    }


}

int main() {

    for (int i = 0; i < sizeof(BLOCKS)/sizeof(int); ++i) {
        printf("\n=== TMA Bandwidth Test [%s] ===\n", DTYPE_NAME);
        printf("Data type size: %zu bytes\n", sizeof(dtype));
        printf("Block size = %d, Threads = %d\n", BLOCKS[i], THREADS_PER_BLOCK);
        printf("Total tile size = %.3f KB (%d bytes)\n", LOAD_SIZE/1024.0, LOAD_SIZE);
        printf("Each thread loads: %d bytes (%d elements)\n\n",
               TMA_BYTES_PER_THREAD, TMA_BYTES_PER_THREAD / (int)sizeof(dtype));

        dtype *dsink = (dtype *)malloc(sizeof(dtype));

        dtype *array_g;
        dtype *dsink_g;

        CUDA_CHECK(cudaMalloc(&array_g, sizeof(dtype) * ARRAY_SIZE));
        CUDA_CHECK(cudaMalloc(&dsink_g, sizeof(dtype)));

        init_data<<<BLOCKS[i], THREADS_PER_BLOCK>>>(array_g);

        auto kernel = &tma_bw;
        int smem_size = LOAD_SIZE;
        if (smem_size >= 48 * 1024) {
            cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
        }
        auto kernel_multi = &tma_bw_multi_thread;
        if (smem_size >= 48 * 1024) {
            cudaFuncSetAttribute(kernel_multi, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
        }

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        // Test single-thread version
        printf("[Single-thread TMA version]\n");
        cudaEventRecord(start);
        kernel<<<BLOCKS[i], THREADS_PER_BLOCK, smem_size>>>(array_g, dsink_g);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        CUDA_CHECK(cudaPeekAtLastError());
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);

        CUDA_CHECK(cudaMemcpy(dsink, dsink_g, sizeof(dtype), cudaMemcpyDeviceToHost));
        printf("Total time = %f ms, transfer size = %lu bytes\n", milliseconds, ARRAY_SIZE * sizeof(dtype));
        printf("Throughput: %.2f GB/s\n\n", ARRAY_SIZE * sizeof(dtype) / (milliseconds / 1000) / 1024 / 1024 / 1024);

        // Test multi-thread version
        printf("[Multi-thread TMA version]\n");
        cudaEventRecord(start);
        kernel_multi<<<BLOCKS[i], THREADS_PER_BLOCK, smem_size>>>(array_g, dsink_g);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        CUDA_CHECK(cudaPeekAtLastError());
        cudaEventElapsedTime(&milliseconds, start, stop);

        CUDA_CHECK(cudaMemcpy(dsink, dsink_g, sizeof(dtype), cudaMemcpyDeviceToHost));
        printf("Total time = %f ms, transfer size = %lu bytes\n", milliseconds, ARRAY_SIZE * sizeof(dtype));
        printf("Throughput: %.2f GB/s\n", ARRAY_SIZE * sizeof(dtype) / (milliseconds / 1000) / 1024 / 1024 / 1024);

        cudaFree(array_g);
        cudaFree(dsink_g);
        free(dsink);
    }


}
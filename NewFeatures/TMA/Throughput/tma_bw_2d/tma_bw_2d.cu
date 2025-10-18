#include <cuda.h>               // CUtensormap
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
    CUtensorMapDataType tm_dtype = CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_FLOAT16;
    #define DTYPE_NAME "FP16"
#elif defined(USE_BFLOAT16)
    typedef __nv_bfloat16 dtype;
    CUtensorMapDataType tm_dtype = CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_BFLOAT16;
    #define DTYPE_NAME "BF16"
#else  // USE_FLOAT32 or default
    typedef float dtype;
    CUtensorMapDataType tm_dtype = CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_FLOAT32;
    #define DTYPE_NAME "FP32"
#endif


#define ARRAY_SIZE (4 * 1024*1024*(1024/sizeof(dtype))) // 4 GB total for all data types
#define GMEM_WIDTH (32*1024)
#define GMEM_HEIGHT (32*1024)
constexpr uint SMEM_WIDTH[] = {64, 64, 64, 64, 64, 64, 128, 128, 256, 256}; // 0.5-128KB
constexpr uint  SMEM_HEIGHT[] = {1, 4, 8, 16, 32, 64, 64, 128, 128, 256}; // sizeof(float) * 2*32 * 2*32 equals to LOAD_SIZE 
// constexpr uint BLOCKS[] = {114, 228, 342, 456};	
constexpr uint BLOCKS[] = {132}; // same as number of SMs, simulate persistent kernels	
#define THREADS_PER_BLOCK 128
constexpr uint IDX = 9;
constexpr uint LOAD_SIZE (SMEM_WIDTH[IDX]*SMEM_HEIGHT[IDX]*sizeof(dtype)); //bytes


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

__global__ void tma_bw_2d(const __grid_constant__ CUtensorMap tma_desc, dtype *dsink)
{

    uint32_t tid = threadIdx.x;
	uint32_t uid = blockIdx.x * blockDim.x + tid;
    // dtype temp_res = 0;

    // __shared__ alignas(16) dtype smem[LOAD_SIZE/sizeof(dtype)];
    // TMA requires 128-byte alignment for shared memory
    extern __shared__ __align__(128) dtype smem[];

#pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ barrier bar;
    if (tid == 0) {
        init(&bar, 1);                    // a) only one thread issues TMA copy, so only wait for 1 thread
        asm volatile("fence.proxy.async.shared::cta;");     // b)
        
        // for (int i = uid; i < ARRAY_SIZE * sizeof(dtype) / LOAD_SIZE; i += gridDim.x * blockDim.x) {
        for (int i = uid; i < ARRAY_SIZE * sizeof(dtype) / LOAD_SIZE; i += gridDim.x * 1) {
            int tensor_cood_x = (i % (GMEM_WIDTH / SMEM_WIDTH[IDX])) * SMEM_WIDTH[IDX];
            int tensor_cood_y = (i / (GMEM_WIDTH / SMEM_WIDTH[IDX])) * SMEM_HEIGHT[IDX];
            asm volatile(
                "{\t\n"
                //"discard.L2 [%1], 128;\n\t"
                "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4]; // 1a. unicast\n\t"
                "mbarrier.expect_tx.relaxed.cta.shared::cta.b64 [%4], %5;\n\t"
                "}"
                :
                //: "r"(static_cast<unsigned>(__cvta_generic_to_shared(ptr))), "l"(ptr[0]), "n"(cuda::aligned_size_t<16>(LOAD_SIZE)), "r"(static_cast<unsigned>(__cvta_generic_to_shared(&bar)))
                : "r"(static_cast<unsigned>(__cvta_generic_to_shared(smem))), "l"(reinterpret_cast<uint64_t>(&tma_desc)), "r"(tensor_cood_x), "r"(tensor_cood_y), "r"(static_cast<unsigned>(__cvta_generic_to_shared(&bar))), "n"(LOAD_SIZE)
                : "memory"); 


            // 3b. All threads arrive on the barrier
            barrier::arrival_token token = bar.arrive();

            // 3c. Wait for the data to have arrived.
            bar.wait(std::move(token));
            //temp_res += smem[0];
        }


    }


}

void create_tensor_map(CUtensorMap & tma_desc, dtype * array)
{
    auto rank = 2;
    uint64_t size[rank] = {GMEM_WIDTH, GMEM_HEIGHT};
    // The stride is the number of bytes to traverse from the first element of one row to the next.
    // It must be a multiple of 16.
    uint64_t stride[rank - 1] = {GMEM_WIDTH * sizeof(dtype)};
    // The box_size is the size of the shared memory buffer that is used as the destination of a TMA transfer.
    uint32_t box_size[rank] = {SMEM_WIDTH[IDX], SMEM_HEIGHT[IDX]};
    // The distance between elements in units of sizeof(element). A stride of 2
    // can be used to load only the real component of a complex-valued tensor, for instance.
    uint32_t elem_stride[rank] = {1, 1};
    // Interleave patterns are sometimes used to accelerate loading of values that
    // are less than 4 bytes long.
    CUtensorMapInterleave interleave = CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE;
    // Swizzling can be used to avoid shared memory bank conflicts.
    CUtensorMapSwizzle swizzle = CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE;
    CUtensorMapL2promotion l2_promotion = CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE;
    // Any element that is outside of bounds will be set to zero by the TMA transfer.
    CUtensorMapFloatOOBfill oob_fill = CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE;

    // Get a function pointer to the cuTensorMapEncodeTiled driver API.
    auto cuTensorMapEncodeTiled = get_cuTensorMapEncodeTiled();

    // Create the tensor descriptor.
    CUresult res = cuTensorMapEncodeTiled(
        &tma_desc,    // CUtensorMap *tensorMap,
        tm_dtype,        // CUtensorMapDataType tensorDataType,
        rank,         // cuuint32_t tensorRank,
        array,       // void *globalAddress,
        size,         // const cuuint64_t *globalDim,
        stride,       // const cuuint64_t *globalStrides,
        box_size,     // const cuuint32_t *boxDim,
        elem_stride,  // const cuuint32_t *elementStrides,
        interleave,   // CUtensorMapInterleave interleave,
        swizzle,      // CUtensorMapSwizzle swizzle,
        l2_promotion, // CUtensorMapL2promotion l2Promotion,
        oob_fill      // CUtensorMapFloatOOBfill oobFill);
    );
    printf("cuTensorMapEncodeTiled returned CUresult: %d\n\n", res);

}

int main() {

    for (int i = 0; i < sizeof(BLOCKS)/sizeof(int); ++i) {
        printf("\n=== TMA 2D Bandwidth Test [%s] ===\n", DTYPE_NAME);
        printf("Data type size: %zu bytes\n", sizeof(dtype));
        printf("Block size = %d, X = %d, Y = %d, Load size = %.2lf KB\n", BLOCKS[i], SMEM_WIDTH[IDX], SMEM_HEIGHT[IDX], float(LOAD_SIZE)/1024);
        dtype *dsink = (dtype *)malloc(sizeof(dtype));

        dtype *array_g;
        dtype *dsink_g;

        CUDA_CHECK(cudaMalloc(&array_g, sizeof(dtype) * ARRAY_SIZE));
        CUDA_CHECK(cudaMalloc(&dsink_g, sizeof(dtype)));

        init_data<<<BLOCKS[i], THREADS_PER_BLOCK>>>(array_g);

        CUtensorMap tma_desc{};
        create_tensor_map(tma_desc, array_g);

        auto kernel = &tma_bw_2d;
        int smem_size = LOAD_SIZE;
        if (smem_size >= 48 * 1024) {
            cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
        }
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);

        kernel<<<BLOCKS[i], THREADS_PER_BLOCK, smem_size>>>(tma_desc, dsink_g);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        CUDA_CHECK(cudaPeekAtLastError());
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);

        CUDA_CHECK(cudaMemcpy(dsink, dsink_g, sizeof(dtype), cudaMemcpyDeviceToHost));
        printf("Total time = %f ms, transfer size = %lu bytes\n", milliseconds, ARRAY_SIZE * sizeof(dtype));
        printf("Throughput: %f GB/s\n", ARRAY_SIZE * sizeof(dtype) / (milliseconds / 1000) / 1024 / 1024 / 1024);
    }
}
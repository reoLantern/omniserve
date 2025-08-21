#include "cuda.h"
#include "cuda_runtime.h"
#include "cuda_fp16.h"

#include "context_pool_kernel.h"
#include "block_info.h"
#include "static_switch.h"
#include "../../../qsrv_trace.h"
#include "../../../call_logger.h"

template <
    typename T_cache,
    bool KV_WITH_ZEROS,
    size_t kBlockM, 
    size_t PoolBlock, 
    size_t HeadDim, 
    size_t WarpSize, 
    size_t PackedNum>
inline __device__ void context_min_max_pool_kernel(
    const Context_pool_params &params, 
    const int bidb, 
    const int bidh, 
    const int m_block
){
    const BlockInfo binfo(params, bidb);
    if (m_block * kBlockM >= binfo.actual_seqlen) return;
    const int tid = threadIdx.x;
    const int BlockSize = blockDim.x;
    const int token_loop_max = kBlockM / BlockSize;     // How many tokens are processed in one thread
    const int channel_loop_max = HeadDim / PackedNum;   // PackedNum = 8
    half min_buffer[PackedNum];
    half max_buffer[PackedNum];
    const int input_hid = params.pooling_heads_idx[bidh];       // Mapping to logic head idx
    # pragma unroll
    for (int token_loop = 0; token_loop < token_loop_max; token_loop++){
        const int logic_token_idx_base = int(m_block * kBlockM + BlockSize * token_loop);
        const int logic_token_idx = logic_token_idx_base + tid;
        if (logic_token_idx_base >= binfo.actual_seqlen){
            break;
        }
        const int token_idx = min(logic_token_idx, binfo.actual_seqlen - 1); // this will not affect min max pooling
        const int page_idx = token_idx / params.page_size;
        const int pool_idx = (token_idx % params.page_size) / params.pooling_size;
        half *input_token_ptr = params.input_ptr + binfo.input_offset(params.input_row_stride) + token_idx * params.input_row_stride + input_hid * params.input_head_stride;
        // add by JXGuo: note this is only right when tcache is int8_t, otherwise you cannot add a byte number
        T_cache *k_cache_batch = reinterpret_cast<T_cache *>(params.kv_buffer.getKBlockPtr(bidb, token_idx));
        half *k_cache_stats_max_ptr = reinterpret_cast<half *>(k_cache_batch + params.kv_buffer.mBytesPerSeq) + params.kv_buffer.mTokensPerBlock * params.pool_h * (KV_WITH_ZEROS? 2 : 1) + bidh * params.input_head_stride;
        half *k_cache_stats_min_ptr = k_cache_stats_max_ptr + params.kv_buffer.SubChunkGroupSize * params.kv_buffer.mElesPerIndicator;

        // const half *output_token_ptr = reinterpret_cast<half*>(params.o_ptrs[bidb * params.o_ptrs_batch_stride + page_idx]) + bidh * params.input_head_stride;
        #pragma unroll
        for (int channel_loop = 0; channel_loop < channel_loop_max; channel_loop++){
            *reinterpret_cast<uint4 *>(min_buffer) = *(reinterpret_cast<uint4 *>(input_token_ptr) + channel_loop);// check here
            // printf("min_buffer[0]: %f, logic_token_idx: %d, hidh: %d, input_hid: %d, bidb: %d\n", __half2float(min_buffer[0]), logic_token_idx, bidh, input_hid, bidb);
            *reinterpret_cast<uint4 *>(max_buffer) = *reinterpret_cast<uint4 *>(min_buffer);
            #pragma unroll
            for (int pack_loop = 0; pack_loop < PackedNum; pack_loop++){
                #pragma unroll
                for (int offset = PoolBlock/2; offset > 0; offset >>= 1){
                    max_buffer[pack_loop] = __hmax(max_buffer[pack_loop], __shfl_xor_sync(0xFFFFFFFF, max_buffer[pack_loop], offset));
                    min_buffer[pack_loop] = __hmin(min_buffer[pack_loop], __shfl_xor_sync(0xFFFFFFFF, min_buffer[pack_loop], offset));
                }
            }
            // __syncthreads();
            if (tid % PoolBlock == 0 && logic_token_idx < binfo.actual_seqlen){
                const int pool_rank_offset = pool_idx * params.pool_h * params.input_head_stride;
                *(reinterpret_cast<uint4 *>(k_cache_stats_max_ptr + pool_rank_offset) + channel_loop) = *reinterpret_cast<uint4 *>(max_buffer);
                // printf("*(half*)((uint4*)(k_cache_stats_max_ptr + pool_rank_offset) + channel_loop): %f\n", __half2float(*(half*)((uint4*)(k_cache_stats_max_ptr + pool_rank_offset) + channel_loop)));

                *(reinterpret_cast<uint4 *>(k_cache_stats_min_ptr + pool_rank_offset) + channel_loop) = *reinterpret_cast<uint4 *>(min_buffer);
            }
        }
    }
}



template <typename T_cache, bool KV_WITH_ZEROS, size_t kBlockM, size_t PoolBlock, size_t HeadDim, size_t WarpSize>
__global__ void context_min_max_pool_compute(Context_pool_params params){
    const int m_block = blockIdx.x;
    // The block index for the batch.
    const int bidb = blockIdx.y;
    // The block index for the head.
    const int bidh = blockIdx.z;

    context_min_max_pool_kernel<T_cache, KV_WITH_ZEROS, kBlockM, PoolBlock, HeadDim, WarpSize, 8/*packed_load fp16*/>(params, bidb, bidh, m_block);
}

template <typename T_cache, bool KV_WITH_ZEROS, size_t PoolBlock, size_t HeadDim>
void launch_context_paged_min_max_pool(const Context_pool_params &params, cudaStream_t stream){
    const int kBlockM = 1024;
    const int BLOCK_SIZE = 256;
    const int num_m_block = (params.max_seqlen_rounded + kBlockM - 1) / kBlockM;
    dim3 grid(num_m_block, params.b, params.pool_h);
    dim3 block(BLOCK_SIZE);
    QSRV_LAUNCH_BEGIN_LIGHT("context_pool", "context_min_max_pool_compute",
                        grid, block, /*smem*/0);
    QSRV_LAUNCH_END();
    context_min_max_pool_compute<T_cache, KV_WITH_ZEROS, kBlockM, PoolBlock, HeadDim, 32><<<grid, block>>>(params);
}



#define CHECK_DEVICE(x) TORCH_CHECK(x.is_cuda(), #x " must be on CUDA")
#define CHECK_SHAPE(x, ...) TORCH_CHECK(x.sizes() == torch::IntArrayRef({__VA_ARGS__}), #x " must have shape (" #__VA_ARGS__ ")")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")



void set_params(
    Context_pool_params &params,
    const size_t batch_size,
    const size_t head_size,
    const size_t max_seqlen_rounded,
    const size_t pooling_size,
    const size_t page_size,
    const size_t input_h,
    const size_t pool_h,
    // device pointers
    const at::Tensor input,
    // const at::Tensor output_ptrs,
    void *cu_seqlens,
    void *pooling_heads_idx,
    KVBlockArray<false> &kv_buffer
){
    // Reset the parameters
    params = {};
    // Set the pointers and strides.
    params.input_ptr = reinterpret_cast<half*>(input.data_ptr());
    // All stride are in elements, not bytes.
    params.input_row_stride = input.stride(-3);
    params.input_head_stride = input.stride(-2);
    // params.o_ptrs = reinterpret_cast<int64_t*>(output_ptrs.data_ptr());
    // params.o_ptrs_batch_stride = output_ptrs.stride(-2);
    params.cu_seqlens = static_cast<int *>(cu_seqlens);
    params.pooling_heads_idx = static_cast<int *>(pooling_heads_idx);

    // Set the dimensions.
    params.b = batch_size;
    params.d = head_size;
    params.max_seqlen_rounded = max_seqlen_rounded;
    params.pooling_size = pooling_size;
    params.page_size = page_size;
    params.input_h = input_h;
    params.pool_h = pool_h;
    params.kv_buffer = kv_buffer;
}


void context_paged_min_max_pool(
    at::Tensor input,              //total_seqlen x num_heads x head_size
    c10::optional<torch::Tensor> _retrieval_kv_pointers, // B x 2 x M
    at::Tensor cu_seqlens,         // b + 1
    at::Tensor pooling_heads_idx,  // num_heads
    const int max_seqlen,
    const int pooling_size,
    const int page_size,
    const int size_per_retrieval_token,   // default = hidden_size * sizeof(dtype)
    const bool kv_cache_with_zeros
){
    QSRV_TRACE_HIT("fused_attention_sparse_utils", "context_paged_min_max_pool");

    TORCH_CHECK(input.dtype() == torch::kFloat16, "context pooling only support fp16 for input");
    // TORCH_CHECK(output_ptrs.dtype() == torch::kInt64, "context pooling only support int32 for output_ptrs");
    TORCH_CHECK(cu_seqlens.dtype() == torch::kInt32, "context pooling only support int32 for cu_seqlens");
    TORCH_CHECK(pooling_heads_idx.dtype() == torch::kInt32, "context pooling only support int32 for pooling_heads_idx");

    CHECK_DEVICE(input);
    // CHECK_DEVICE(output_ptrs);
    CHECK_DEVICE(cu_seqlens);
    CHECK_DEVICE(pooling_heads_idx);

    CHECK_CONTIGUOUS(input);
    // CHECK_CONTIGUOUS(output_ptrs);
    CHECK_CONTIGUOUS(cu_seqlens);
    CHECK_CONTIGUOUS(pooling_heads_idx);
    // printf("start");
    const int batch_size = cu_seqlens.numel() - 1;
    const int head_size = input.sizes()[2];
    const int num_input_heads = input.sizes()[1];
    const int num_pooling_heads = pooling_heads_idx.numel();
    const int max_seqlen_rounded = int((max_seqlen + pooling_size - 1) / pooling_size) * pooling_size;

    QSRV_CALL_BEGIN("context_pool", "context_paged_min_max_pool");
    QSRV_ARG_TENSOR("input", input);
    QSRV_ARG_OPT_TENSOR("retrieval_kv_ptrs", _retrieval_kv_pointers);
    QSRV_ARG_TENSOR("cu_seqlens", cu_seqlens);
    QSRV_ARG_TENSOR("pooling_heads_idx", pooling_heads_idx);
    QSRV_ARG_I("max_seqlen", max_seqlen);
    QSRV_ARG_I("pooling_size", pooling_size);
    QSRV_ARG_I("page_size", page_size);
    QSRV_ARG_I("size_per_retrieval_token", size_per_retrieval_token);
    QSRV_ARG_I("batch_size", batch_size);
    QSRV_ARG_I("head_size", head_size);
    QSRV_ARG_I("num_input_heads", num_input_heads);
    QSRV_ARG_I("num_pooling_heads", num_pooling_heads);
    QSRV_ARG_I("max_seqlen_rounded", max_seqlen_rounded);
    QSRV_ARG_B("kv_cache_with_zeros", kv_cache_with_zeros);
    QSRV_CALL_END();

    int retrieval_max_blocks_per_seq = 0;
    if (_retrieval_kv_pointers.has_value()) {
        auto retrieval_kv_pointers = _retrieval_kv_pointers.value();
        retrieval_max_blocks_per_seq = retrieval_kv_pointers.size(-1);
        CHECK_DEVICE(retrieval_kv_pointers);
        CHECK_CONTIGUOUS(retrieval_kv_pointers);
    }

    KVBlockArray<false> retrieval_kv_buffer(batch_size, retrieval_max_blocks_per_seq, page_size, size_per_retrieval_token, 0, 0, 0, 0, pooling_size, num_pooling_heads * head_size);
    retrieval_kv_buffer.data = _retrieval_kv_pointers.has_value() ? _retrieval_kv_pointers.value().data_ptr<int64_t>() : nullptr;


    Context_pool_params params;
    set_params(params,
        batch_size,
        head_size,
        max_seqlen_rounded,
        pooling_size,
        page_size,
        num_input_heads,
        num_pooling_heads,
        input,
        // output_ptrs,
        cu_seqlens.data_ptr(),
        pooling_heads_idx.data_ptr(),
        retrieval_kv_buffer
    );

    auto stream = at::cuda::getCurrentCUDAStream();
    POOL_SWITCH(pooling_size, [&] {
        HEADDIM_SWITCH(head_size, [&] {
            KVZERO_SWITCH(kv_cache_with_zeros, [&] {
              launch_context_paged_min_max_pool<int8_t/*T_cache*/, EnableZero, PoolBlock, HeadDim>(params, stream);  
            });
        });
    });
}
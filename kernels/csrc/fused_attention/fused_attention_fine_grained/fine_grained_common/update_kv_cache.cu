// Inspired by TRT-LLM.
// Modified by Haotian Tang and Shang Yang.
// @article{lin2024qserve,
//   title={QServe: W4A8KV4 Quantization and System Co-design for Efficient LLM Serving},
//   author={Lin*, Yujun and Tang*, Haotian and Yang*, Shang and Zhang, Zhekai and Xiao, Guangxuan and Gan, Chuang and Han, Song},
//   journal={arXiv preprint arXiv:2405.04532},
//   year={2024}
// }
// @article{yang2025lserve,
//   title={LServe: Efficient Long-sequence LLM Serving with Unified Sparse Attention},
//   author={Yang*, Shang and Guo*, Junxian and Tang, Haotian and Hu, Qinghao and Xiao, Guangxuan and Tang, Jiaming and Lin, Yujun and Liu, Zhijian and Lu, Yao and Han, Song},
//   year={2025}
// }
#include <torch/extension.h>
#include "ATen/cuda/CUDAContext.h"
#include <c10/cuda/CUDAGuard.h>

#include "applyBiasRopeUpdateKVCache.h"
#include "../../../qsrv_trace.h"
#include "../../../call_logger.h"

INSTANTIATE_ADDFUSEDQKVBIAS_TRANSPOSE(half, KVBlockArray<false>, KVBlockArray<true>, true);
INSTANTIATE_ADDFUSEDQKVBIAS_TRANSPOSE(half, KVBlockArray<false>, KVBlockArray<true>, false);
// INSTANTIATE_ADDFUSEDQKVBIAS_TRANSPOSE(half, KVBlockArray, KVBlockArray, true);
// INSTANTIATE_ADDFUSEDQKVBIAS_TRANSPOSE(half, KVBlockArray, KVBlockArray, false);
// INSTANTIATE_ADDFUSEDQKVBIAS_TRANSPOSE(half, KVLinearBuffer, KVLinearBuffer, true);
// INSTANTIATE_ADDFUSEDQKVBIAS_TRANSPOSE(half, KVLinearBuffer, KVLinearBuffer, false);

void apply_bias_rope_update_kv_cache(const torch::Tensor qkv,
                                     torch::Tensor retrieval_seq_lens,
                                     c10::optional<torch::Tensor> streaming_seq_lens,
                                     torch::Tensor padding_offset,
                                    //  c10::optional<torch::Tensor> kv_pointers, // B x 2 x M
                                     c10::optional<torch::Tensor> retrieval_kv_pointers, // B x 2 x M
                                     c10::optional<torch::Tensor> streaming_kv_pointers, // B x 2 x M
                                     torch::Tensor retrieval_head_flags, // H
                                     torch::Tensor head_rank_table, // H
                                     // virtual sequence length (after padding)
                                     const int head_num,
                                     const int kv_head_num,
                                     const int seq_len,          // max seq len
                                     const int tokens_per_block, // default=64
                                    //  const int size_per_token,   // default = hidden_size * sizeof(dtype)
                                     const int size_per_retrieval_token,   // default = hidden_size * sizeof(dtype)
                                     const int size_per_streaming_token,   // default = hidden_size * sizeof(dtype)
                                     const int sink_token_num, const int local_token_num,
                                     const int sink_block_num, const int local_block_num,
                                     const int num_retrieval_kv_heads,
                                     const int num_streaming_kv_heads,
                                     const int rotary_embedding_dim,
                                     const float rotary_embedding_base,
                                     const float rotary_embedding_scale,
                                     const int rotary_embedding_max_positions,
                                     // neox_rotary_style = not interleaved
                                     const bool neox_rotary_style,
                                     const bool int4_kv_cache,
                                     const bool kv_cache_with_zeros
                                     )
{
    QSRV_TRACE_HIT("fused_attention_fine_grained", "apply_bias_rope_update_kv_cache");

    QSRV_CALL_BEGIN("fused_attention_fine_grained", "apply_bias_rope_update_kv_cache");
    QSRV_ARG_TENSOR("qkv", qkv);
    QSRV_ARG_TENSOR("retrieval_seq_lens", retrieval_seq_lens);
    QSRV_ARG_OPT_TENSOR("streaming_seq_lens",      streaming_seq_lens);
    QSRV_ARG_OPT_TENSOR("retrieval_kv_pointers",   retrieval_kv_pointers);
    QSRV_ARG_OPT_TENSOR("streaming_kv_pointers",   streaming_kv_pointers);
    QSRV_ARG_TENSOR("retrieval_head_flags", retrieval_head_flags);
    QSRV_ARG_TENSOR("head_rank_table",      head_rank_table);
    QSRV_ARG_I("head_num",                  head_num);
    QSRV_ARG_I("kv_head_num",               kv_head_num);
    QSRV_ARG_I("seq_len",                   seq_len);
    QSRV_ARG_I("tokens_per_block",          tokens_per_block);
    QSRV_ARG_I("size_per_retrieval_token",  size_per_retrieval_token);
    QSRV_ARG_I("size_per_streaming_token",  size_per_streaming_token);
    QSRV_ARG_I("sink_token_num",            sink_token_num);
    QSRV_ARG_I("local_token_num",           local_token_num);
    QSRV_ARG_I("sink_block_num",            sink_block_num);
    QSRV_ARG_I("local_block_num",           local_block_num);
    QSRV_ARG_I("num_retrieval_kv_heads",    num_retrieval_kv_heads);
    QSRV_ARG_I("num_streaming_kv_heads",    num_streaming_kv_heads);
    QSRV_ARG_I("rotary_embedding_dim",      rotary_embedding_dim);
    QSRV_ARG_F("rotary_embedding_base",     rotary_embedding_base);
    QSRV_ARG_F("rotary_embedding_scale",    rotary_embedding_scale);
    QSRV_ARG_I("rotary_embedding_max_positions", rotary_embedding_max_positions);
    QSRV_ARG_B("neox_rotary_style",         neox_rotary_style);
    QSRV_ARG_B("int4_kv_cache",             int4_kv_cache);
    QSRV_ARG_B("kv_cache_with_zeros",       kv_cache_with_zeros);
    QSRV_CALL_END();
    half *q_ptr = nullptr;
    half *qkv_ptr = reinterpret_cast<half *>(qkv.data_ptr<at::Half>());
    int *retrieval_seq_lens_ptr = retrieval_seq_lens.data_ptr<int>();
    int *retrieval_kv_seq_lens_ptr = retrieval_seq_lens_ptr;
    // TBD
    int *padding_offset_ptr = padding_offset.data_ptr<int>();
    half *qkv_bias_ptr = nullptr;
    int batch_size = retrieval_seq_lens.size(0);
    int retrieval_max_blocks_per_seq = retrieval_kv_pointers.has_value() ? retrieval_kv_pointers.value().size(-1) : 0;
    int streaming_max_blocks_per_seq = streaming_kv_pointers.has_value() ? streaming_kv_pointers.value().size(-1) : 0;
    KVBlockArray<false> retrievalkvTable(batch_size, retrieval_max_blocks_per_seq, tokens_per_block, size_per_retrieval_token, 0, 0, 0, 0, 0, 0);       // NOTE (Shang): The last two args are for hierarchical paging. Not sure if the current impl is correct. 
    KVBlockArray<true> streamingkvTable(batch_size, streaming_max_blocks_per_seq, tokens_per_block, size_per_streaming_token, sink_token_num, local_token_num, sink_block_num, local_block_num, 0, 0);
    
    retrievalkvTable.data = retrieval_kv_pointers.has_value() ? retrieval_kv_pointers.value().data_ptr<int64_t>() : nullptr;
    streamingkvTable.data = streaming_kv_pointers.has_value() ? streaming_kv_pointers.value().data_ptr<int64_t>() : nullptr;

    int *retrieval_head_flags_ptr = retrieval_head_flags.data_ptr<int>();
    int *head_rank_table_ptr = head_rank_table.data_ptr<int>();

    // NOTE: cyclic_kv_cache_len should not be 0.
    int cyclic_kv_cache_len = rotary_embedding_max_positions;
    int sink_token_len = 0;
    int token_num = qkv.size(0);
    // fix this
    int size_per_head = rotary_embedding_dim;
    RotaryScalingType rotary_scale_type = RotaryScalingType::kLINEAR;
    PositionEmbeddingType position_embedding_type = PositionEmbeddingType::kROPE_GPT_NEOX;
    int *medusa_position_offsets_ptr = nullptr;
    bool position_shift_enabled = false;
    float *scale_ptr = nullptr;
    int int8_mode = 1;
    KvCacheDataType cache_type;
    if (int4_kv_cache) {
        if (kv_cache_with_zeros)
        {
            cache_type = KvCacheDataType::ZINT4;
        }
        else
        {
            cache_type = KvCacheDataType::INT4;
        }
    }
    else {
        if (kv_cache_with_zeros)
        {
            cache_type = KvCacheDataType::ZINT8;
        }
        else
        {
            cache_type = KvCacheDataType::INT8;
        }
    }
    int beam_width = 1;
    bool enable_paged_kv_fmha = true;
    // TODO: grid_block_cache for different devices??
    int2 grid_block_cache = make_int2(96, 1024);
    auto stream = at::cuda::getCurrentCUDAStream();
    invokeApplyBiasRopeUpdateKVCache<half, KVBlockArray<false>, KVBlockArray<true>, false>(
        qkv_ptr, q_ptr, retrievalkvTable, streamingkvTable, retrieval_head_flags_ptr, head_rank_table_ptr, 
        qkv_bias_ptr, retrieval_seq_lens_ptr, retrieval_kv_seq_lens_ptr,
        padding_offset_ptr, batch_size, seq_len, cyclic_kv_cache_len,
        sink_token_len, token_num, head_num, kv_head_num, num_retrieval_kv_heads, num_streaming_kv_heads,
        size_per_head, rotary_embedding_dim, rotary_embedding_base,
        rotary_scale_type, rotary_embedding_scale, rotary_embedding_max_positions,
        position_embedding_type, medusa_position_offsets_ptr, position_shift_enabled,
        scale_ptr, int8_mode, cache_type,
        enable_paged_kv_fmha, beam_width, grid_block_cache, stream);

    // void invokeApplyBiasRopeUpdateKVCache<T, KVCacheBuffer, IS_GENERATE>(T * QKV, T * Q,
    //     KVCacheBuffer & kvTable, const T* qkv_bias, const int* seq_lens, const int* kv_seq_lens,
    //     const int* padding_offset, const int batch_size, const int seq_len, const int cyclic_kv_cache_len,
    //     const int sink_token_len, const int token_num, const int head_num, const int kv_head_num,
    //     const int size_per_head, const int rotary_embedding_dim, const float rotary_embedding_base,
    //     const RotaryScalingType rotary_scale_type, const float rotary_embedding_scale,
    //     const int rotary_embedding_max_positions, const PositionEmbeddingType position_embedding_type,
    //     const int* medusa_position_offsets, const bool position_shift_enabled, const float* scale,
    //     const int int8_mode, const KvCacheDataType cache_type, const float* kvScaleOrigQuant,
    //     const bool enable_paged_kv_fmha, const int beam_width, int2& grid_block_cache, cudaStream_t stream);
}
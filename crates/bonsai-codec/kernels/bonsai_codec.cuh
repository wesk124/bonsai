#pragma once
#include <stdint.h>
typedef struct {
  uint64_t k_ptr, v_ptr;
  uint64_t stride_head, stride_token, stride_dim;
  uint32_t num_kv_heads, head_dim;
  uint32_t layer_begin, layer_end, token_begin, token_end;
  uint64_t frame, k_payload, v_payload, k_meta, v_meta;
  uint32_t bits, k_group, v_group, k_axis, v_axis, k_symmetric, v_symmetric;
} BonsaiChunkArgs;
#ifdef __cplusplus
extern "C" {
#endif
int bonsai_validate_chunk(const BonsaiChunkArgs*, uint64_t stream);
int bonsai_validate_metadata(const BonsaiChunkArgs*, uint64_t stream);
int bonsai_zero_frame(uint64_t frame, uint64_t bytes, uint64_t stream);
int bonsai_encode_chunk(const BonsaiChunkArgs*, uint64_t stream);
int bonsai_decode_chunk(const BonsaiChunkArgs*, uint64_t stream);
#ifdef __cplusplus
}
#endif

#ifndef H3_TOKEN_IDS_H
#define H3_TOKEN_IDS_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* G1 product path (CUDA): load Qwen token ids without Metal ObjC tokenizer.
 * Full HF tokenizer.json BPE remains Metal-only; CUDA callers feed ids from
 * file / canary / comma list. Caller frees *ids with free(). */

int h3_token_ids_load_file(const char *path, int32_t **ids, size_t *count,
                           char *error, size_t error_size);

/* Parse "1,2,3" or whitespace-separated ints. */
int h3_token_ids_parse_list(const char *text, int32_t **ids, size_t *count,
                            char *error, size_t error_size);

/* Built-in canaries matching tests/test_tokenizer.c (released Qwen vocab).
 * name: "fox" | "cinematic" | "im_start" | NULL→fox */
int h3_token_ids_canary(const char *name, int32_t **ids, size_t *count,
                        char *error, size_t error_size);

/* Pad or truncate to exactly want tokens (PAD=151643). */
int h3_token_ids_fit(int32_t **ids, size_t *count, size_t want, char *error,
                     size_t error_size);

#ifdef __cplusplus
}
#endif

#endif

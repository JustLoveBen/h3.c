#ifndef H3_TE_CACHE_H
#define H3_TE_CACHE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Opt-in TE prompt embedding disk cache. Gold default is off.
 * Key = token ids + te_layers + TE path fingerprint (mtime+size).
 * Vision / multimodal embeddings are never cached. */

enum { H3_TE_CACHE_TEXT_DIM = 5120 };

int h3_te_cache_key(const int32_t *ids, size_t n, unsigned te_layers,
                    const char *te_path, char *hex, size_t hex_size,
                    char *err, size_t err_size);
int h3_te_cache_load(const char *dir, const char *hex, size_t tokens,
                     unsigned te_layers, float **out, char *err,
                     size_t err_size);
int h3_te_cache_save(const char *dir, const char *hex, const float *emb,
                     size_t tokens, unsigned te_layers, char *err,
                     size_t err_size);

#ifdef __cplusplus
}
#endif

#endif

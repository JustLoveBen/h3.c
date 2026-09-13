#ifndef H3_ULYSSES_H
#define H3_ULYSSES_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

enum { H3_ULYSSES_PACKED_ALIGN = 64 };

typedef struct h3_ulysses_cfg {
    int tp_size; /* P1: 1 */
    int ulysses; /* 1,2,4,8 */
    int ring;    /* 1 */
    int rank;
    int world_size;
    void *comm; /* owned by h3_ulysses_comm_*; NULL when U=1 */
} h3_ulysses_cfg;

int h3_ulysses_validate(int heads, int packed_align, int tp, int ulysses,
                        int ring, char *err, size_t err_size);
size_t h3_ulysses_pad_seq(size_t seq, int align);
size_t h3_ulysses_local_rows(size_t seq_pad, int ulysses, int rank);

#ifdef __cplusplus
}
#endif

#endif

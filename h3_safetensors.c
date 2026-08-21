#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "h3_safetensors.h"

#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#define H3_ST_MAX_HEADER (256u * 1024u * 1024u)

typedef struct {
    const char *at;
    const char *end;
    char *error;
    size_t error_size;
} h3_json_cursor;

static int h3_json_fail(h3_json_cursor *cursor, const char *message) {
    if (cursor->error && cursor->error_size) {
        snprintf(cursor->error, cursor->error_size, "%s", message);
    }
    return 0;
}

static void h3_json_ws(h3_json_cursor *cursor) {
    while (cursor->at < cursor->end &&
           (*cursor->at == ' ' || *cursor->at == '\n' ||
            *cursor->at == '\r' || *cursor->at == '\t')) cursor->at++;
}

static int h3_json_take(h3_json_cursor *cursor, char expected) {
    h3_json_ws(cursor);
    if (cursor->at >= cursor->end || *cursor->at != expected) {
        return h3_json_fail(cursor, "malformed safetensors JSON header");
    }
    cursor->at++;
    return 1;
}

static int h3_hex(char value) {
    if (value >= '0' && value <= '9') return value - '0';
    if (value >= 'a' && value <= 'f') return value - 'a' + 10;
    if (value >= 'A' && value <= 'F') return value - 'A' + 10;
    return -1;
}

static char *h3_json_string(h3_json_cursor *cursor) {
    h3_json_ws(cursor);
    if (cursor->at >= cursor->end || *cursor->at != '"') {
        h3_json_fail(cursor, "expected JSON string");
        return NULL;
    }
    cursor->at++;
    size_t maximum = (size_t)(cursor->end - cursor->at);
    char *result = malloc(maximum + 1);
    if (!result) {
        h3_json_fail(cursor, "out of memory parsing safetensors header");
        return NULL;
    }
    size_t length = 0;
    while (cursor->at < cursor->end && *cursor->at != '"') {
        unsigned char value = (unsigned char)*cursor->at++;
        if (value == '\\') {
            if (cursor->at >= cursor->end) goto malformed;
            value = (unsigned char)*cursor->at++;
            switch (value) {
                case '"': result[length++] = '"'; break;
                case '\\': result[length++] = '\\'; break;
                case '/': result[length++] = '/'; break;
                case 'b': result[length++] = '\b'; break;
                case 'f': result[length++] = '\f'; break;
                case 'n': result[length++] = '\n'; break;
                case 'r': result[length++] = '\r'; break;
                case 't': result[length++] = '\t'; break;
                case 'u': {
                    if (cursor->end - cursor->at < 4) goto malformed;
                    int codepoint = 0;
                    for (int index = 0; index < 4; index++) {
                        int digit = h3_hex(cursor->at[index]);
                        if (digit < 0) goto malformed;
                        codepoint = codepoint * 16 + digit;
                    }
                    cursor->at += 4;
                    if (codepoint < 0x80) {
                        result[length++] = (char)codepoint;
                    } else if (codepoint < 0x800) {
                        result[length++] = (char)(0xc0 | (codepoint >> 6));
                        result[length++] = (char)(0x80 | (codepoint & 0x3f));
                    } else {
                        result[length++] = (char)(0xe0 | (codepoint >> 12));
                        result[length++] = (char)(0x80 | ((codepoint >> 6) & 0x3f));
                        result[length++] = (char)(0x80 | (codepoint & 0x3f));
                    }
                    break;
                }
                default: goto malformed;
            }
        } else {
            if (value < 0x20) goto malformed;
            result[length++] = (char)value;
        }
    }
    if (cursor->at >= cursor->end) goto malformed;
    cursor->at++;
    result[length] = '\0';
    return result;

malformed:
    free(result);
    h3_json_fail(cursor, "malformed JSON string escape");
    return NULL;
}

static int h3_json_uint(h3_json_cursor *cursor, uint64_t *result) {
    h3_json_ws(cursor);
    if (cursor->at >= cursor->end || *cursor->at < '0' || *cursor->at > '9') {
        return h3_json_fail(cursor, "expected unsigned JSON integer");
    }
    uint64_t value = 0;
    while (cursor->at < cursor->end && *cursor->at >= '0' && *cursor->at <= '9') {
        unsigned digit = (unsigned)(*cursor->at - '0');
        if (value > (UINT64_MAX - digit) / 10) {
            return h3_json_fail(cursor, "integer overflow in safetensors header");
        }
        value = value * 10 + digit;
        cursor->at++;
    }
    *result = value;
    return 1;
}

static int h3_json_skip(h3_json_cursor *cursor);

static int h3_json_skip_compound(h3_json_cursor *cursor, char open, char close) {
    if (!h3_json_take(cursor, open)) return 0;
    h3_json_ws(cursor);
    if (cursor->at < cursor->end && *cursor->at == close) {
        cursor->at++;
        return 1;
    }
    for (;;) {
        if (open == '{') {
            char *key = h3_json_string(cursor);
            if (!key) return 0;
            free(key);
            if (!h3_json_take(cursor, ':')) return 0;
        }
        if (!h3_json_skip(cursor)) return 0;
        h3_json_ws(cursor);
        if (cursor->at >= cursor->end) return h3_json_fail(cursor, "unterminated JSON value");
        if (*cursor->at == close) {
            cursor->at++;
            return 1;
        }
        if (*cursor->at++ != ',') return h3_json_fail(cursor, "expected JSON comma");
    }
}

static int h3_json_skip(h3_json_cursor *cursor) {
    h3_json_ws(cursor);
    if (cursor->at >= cursor->end) return h3_json_fail(cursor, "missing JSON value");
    if (*cursor->at == '"') {
        char *value = h3_json_string(cursor);
        if (!value) return 0;
        free(value);
        return 1;
    }
    if (*cursor->at == '{') return h3_json_skip_compound(cursor, '{', '}');
    if (*cursor->at == '[') return h3_json_skip_compound(cursor, '[', ']');
    const char *start = cursor->at;
    while (cursor->at < cursor->end && *cursor->at != ',' &&
           *cursor->at != '}' && *cursor->at != ']' &&
           *cursor->at != ' ' && *cursor->at != '\n' &&
           *cursor->at != '\r' && *cursor->at != '\t') cursor->at++;
    if (cursor->at == start) return h3_json_fail(cursor, "invalid JSON scalar");
    return 1;
}

static h3_dtype h3_parse_dtype(const char *name) {
    if (!strcmp(name, "BOOL")) return H3_DTYPE_BOOL;
    if (!strcmp(name, "I8")) return H3_DTYPE_I8;
    if (!strcmp(name, "U8")) return H3_DTYPE_U8;
    if (!strcmp(name, "I16")) return H3_DTYPE_I16;
    if (!strcmp(name, "U16")) return H3_DTYPE_U16;
    if (!strcmp(name, "F16")) return H3_DTYPE_F16;
    if (!strcmp(name, "BF16")) return H3_DTYPE_BF16;
    if (!strcmp(name, "I32")) return H3_DTYPE_I32;
    if (!strcmp(name, "U32")) return H3_DTYPE_U32;
    if (!strcmp(name, "F32")) return H3_DTYPE_F32;
    if (!strcmp(name, "I64")) return H3_DTYPE_I64;
    if (!strcmp(name, "U64")) return H3_DTYPE_U64;
    if (!strcmp(name, "F64")) return H3_DTYPE_F64;
    if (!strcmp(name, "F8_E4M3")) return H3_DTYPE_F8_E4M3;
    return H3_DTYPE_UNKNOWN;
}

size_t h3_dtype_size(h3_dtype dtype) {
    switch (dtype) {
        case H3_DTYPE_BOOL:
        case H3_DTYPE_I8:
        case H3_DTYPE_U8:
        case H3_DTYPE_F8_E4M3: return 1;
        case H3_DTYPE_I16:
        case H3_DTYPE_U16:
        case H3_DTYPE_F16:
        case H3_DTYPE_BF16: return 2;
        case H3_DTYPE_I32:
        case H3_DTYPE_U32:
        case H3_DTYPE_F32: return 4;
        case H3_DTYPE_I64:
        case H3_DTYPE_U64:
        case H3_DTYPE_F64: return 8;
        case H3_DTYPE_UNKNOWN: return 0;
    }
    return 0;
}

const char *h3_dtype_name(h3_dtype dtype) {
    switch (dtype) {
        case H3_DTYPE_BOOL: return "BOOL";
        case H3_DTYPE_I8: return "I8";
        case H3_DTYPE_U8: return "U8";
        case H3_DTYPE_I16: return "I16";
        case H3_DTYPE_U16: return "U16";
        case H3_DTYPE_F16: return "F16";
        case H3_DTYPE_BF16: return "BF16";
        case H3_DTYPE_I32: return "I32";
        case H3_DTYPE_U32: return "U32";
        case H3_DTYPE_F32: return "F32";
        case H3_DTYPE_I64: return "I64";
        case H3_DTYPE_U64: return "U64";
        case H3_DTYPE_F64: return "F64";
        case H3_DTYPE_F8_E4M3: return "F8_E4M3";
        case H3_DTYPE_UNKNOWN: return "UNKNOWN";
    }
    return "UNKNOWN";
}

static int h3_parse_shape(h3_json_cursor *cursor, h3_st_tensor *tensor) {
    if (!h3_json_take(cursor, '[')) return 0;
    h3_json_ws(cursor);
    if (cursor->at < cursor->end && *cursor->at == ']') {
        cursor->at++;
        tensor->ndim = 0;
        return 1;
    }
    int dimensions = 0;
    for (;;) {
        uint64_t value;
        if (dimensions >= 8) return h3_json_fail(cursor, "tensor rank exceeds 8");
        if (!h3_json_uint(cursor, &value)) return 0;
        tensor->shape[dimensions++] = value;
        h3_json_ws(cursor);
        if (cursor->at >= cursor->end) return h3_json_fail(cursor, "unterminated shape");
        if (*cursor->at == ']') {
            cursor->at++;
            tensor->ndim = dimensions;
            return 1;
        }
        if (*cursor->at++ != ',') return h3_json_fail(cursor, "expected shape comma");
    }
}

static int h3_parse_offsets(h3_json_cursor *cursor, h3_st_tensor *tensor) {
    if (!h3_json_take(cursor, '[') ||
        !h3_json_uint(cursor, &tensor->data_begin) ||
        !h3_json_take(cursor, ',') ||
        !h3_json_uint(cursor, &tensor->data_end) ||
        !h3_json_take(cursor, ']')) return 0;
    return 1;
}

static int h3_parse_tensor(h3_json_cursor *cursor, h3_st_tensor *tensor) {
    if (!h3_json_take(cursor, '{')) return 0;
    int have_dtype = 0;
    int have_shape = 0;
    int have_offsets = 0;
    for (;;) {
        h3_json_ws(cursor);
        if (cursor->at < cursor->end && *cursor->at == '}') {
            cursor->at++;
            break;
        }
        char *key = h3_json_string(cursor);
        if (!key || !h3_json_take(cursor, ':')) {
            free(key);
            return 0;
        }
        int ok = 1;
        if (!strcmp(key, "dtype")) {
            char *value = h3_json_string(cursor);
            if (!value) ok = 0;
            else {
                tensor->dtype = h3_parse_dtype(value);
                have_dtype = tensor->dtype != H3_DTYPE_UNKNOWN;
            }
            free(value);
        } else if (!strcmp(key, "shape")) {
            ok = h3_parse_shape(cursor, tensor);
            have_shape = ok;
        } else if (!strcmp(key, "data_offsets")) {
            ok = h3_parse_offsets(cursor, tensor);
            have_offsets = ok;
        } else {
            ok = h3_json_skip(cursor);
        }
        free(key);
        if (!ok) return 0;
        h3_json_ws(cursor);
        if (cursor->at >= cursor->end) return h3_json_fail(cursor, "unterminated tensor entry");
        if (*cursor->at == '}') continue;
        if (*cursor->at++ != ',') return h3_json_fail(cursor, "expected tensor comma");
    }
    if (!have_dtype || !have_shape || !have_offsets) {
        return h3_json_fail(cursor, "incomplete tensor descriptor");
    }
    return 1;
}

static int h3_validate_tensor(h3_json_cursor *cursor, h3_st_tensor *tensor,
                              uint64_t data_start, uint64_t file_size) {
    if (tensor->data_end < tensor->data_begin ||
        tensor->data_end > file_size - data_start) {
        return h3_json_fail(cursor, "tensor data offsets exceed file");
    }
    uint64_t elements = 1;
    for (int dimension = 0; dimension < tensor->ndim; dimension++) {
        if (tensor->shape[dimension] && elements > UINT64_MAX / tensor->shape[dimension]) {
            return h3_json_fail(cursor, "tensor element count overflow");
        }
        elements *= tensor->shape[dimension];
    }
    size_t item_size = h3_dtype_size(tensor->dtype);
    if (item_size && elements > UINT64_MAX / item_size) {
        return h3_json_fail(cursor, "tensor byte count overflow");
    }
    uint64_t expected = elements * item_size;
    if (expected != tensor->data_end - tensor->data_begin) {
        return h3_json_fail(cursor, "tensor shape does not match byte offsets");
    }
    tensor->file_offset = data_start + tensor->data_begin;
    return 1;
}

static int h3_append_tensor(h3_st_header *header, h3_st_tensor tensor,
                            size_t *capacity, h3_json_cursor *cursor) {
    if (header->tensor_count == *capacity) {
        size_t next = *capacity ? *capacity * 2 : 64;
        h3_st_tensor *values = realloc(header->tensors, next * sizeof(*values));
        if (!values) return h3_json_fail(cursor, "out of memory indexing tensors");
        header->tensors = values;
        *capacity = next;
    }
    header->tensors[header->tensor_count++] = tensor;
    return 1;
}

static uint64_t h3_u64_le(const unsigned char bytes[8]) {
    uint64_t value = 0;
    for (unsigned index = 0; index < 8; index++) {
        value |= (uint64_t)bytes[index] << (index * 8);
    }
    return value;
}

int h3_st_read_header(const char *path, h3_st_header *header,
                      char *error, size_t error_size) {
    if (!path || !header) return 0;
    memset(header, 0, sizeof(*header));
    if (error && error_size) error[0] = '\0';
    int descriptor = open(path, O_RDONLY);
    if (descriptor < 0) {
        if (error && error_size) snprintf(error, error_size, "%s: %s", path, strerror(errno));
        return 0;
    }
    struct stat status;
    unsigned char prefix[8];
    if (fstat(descriptor, &status) != 0 || status.st_size < 8 ||
        pread(descriptor, prefix, sizeof(prefix), 0) != (ssize_t)sizeof(prefix)) {
        if (error && error_size) snprintf(error, error_size, "%s: invalid file", path);
        close(descriptor);
        return 0;
    }
    uint64_t file_size = (uint64_t)status.st_size;
    uint64_t header_size = h3_u64_le(prefix);
    if (header_size > H3_ST_MAX_HEADER || header_size > file_size - 8) {
        if (error && error_size) snprintf(error, error_size, "%s: invalid header size", path);
        close(descriptor);
        return 0;
    }
    char *json = malloc((size_t)header_size + 1);
    if (!json || pread(descriptor, json, (size_t)header_size, 8) != (ssize_t)header_size) {
        if (error && error_size) snprintf(error, error_size, "%s: cannot read header", path);
        free(json);
        close(descriptor);
        return 0;
    }
    close(descriptor);
    json[header_size] = '\0';
    header->path = strdup(path);
    header->file_size = file_size;
    header->header_size = header_size;
    if (!header->path) {
        free(json);
        if (error && error_size) snprintf(error, error_size, "out of memory");
        return 0;
    }
    h3_json_cursor cursor = {json, json + header_size, error, error_size};
    size_t capacity = 0;
    if (!h3_json_take(&cursor, '{')) goto fail;
    h3_json_ws(&cursor);
    if (cursor.at < cursor.end && *cursor.at == '}') cursor.at++;
    else for (;;) {
        char *name = h3_json_string(&cursor);
        if (!name || !h3_json_take(&cursor, ':')) {
            free(name);
            goto fail;
        }
        if (!strcmp(name, "__metadata__")) {
            free(name);
            h3_json_ws(&cursor);
            const char *meta_start = cursor.at;
            if (!h3_json_skip(&cursor)) goto fail;
            size_t meta_len = (size_t)(cursor.at - meta_start);
            free(header->metadata_json);
            header->metadata_json = malloc(meta_len + 1);
            if (!header->metadata_json) {
                h3_json_fail(&cursor, "out of memory capturing metadata");
                goto fail;
            }
            memcpy(header->metadata_json, meta_start, meta_len);
            header->metadata_json[meta_len] = '\0';
        } else {
            h3_st_tensor tensor;
            memset(&tensor, 0, sizeof(tensor));
            tensor.name = name;
            if (!h3_parse_tensor(&cursor, &tensor) ||
                !h3_validate_tensor(&cursor, &tensor, 8 + header_size, file_size) ||
                !h3_append_tensor(header, tensor, &capacity, &cursor)) {
                free(tensor.name);
                goto fail;
            }
        }
        h3_json_ws(&cursor);
        if (cursor.at >= cursor.end) {
            h3_json_fail(&cursor, "unterminated safetensors object");
            goto fail;
        }
        if (*cursor.at == '}') {
            cursor.at++;
            break;
        }
        if (*cursor.at++ != ',') {
            h3_json_fail(&cursor, "expected top-level comma");
            goto fail;
        }
    }
    h3_json_ws(&cursor);
    if (cursor.at != cursor.end) {
        h3_json_fail(&cursor, "trailing data in safetensors header");
        goto fail;
    }
    free(json);
    return 1;

fail:
    free(json);
    h3_st_free_header(header);
    return 0;
}

void h3_st_free_header(h3_st_header *header) {
    if (!header) return;
    for (size_t index = 0; index < header->tensor_count; index++) {
        free(header->tensors[index].name);
    }
    free(header->tensors);
    free(header->path);
    free(header->metadata_json);
    memset(header, 0, sizeof(*header));
}

const h3_st_tensor *h3_st_find(const h3_st_header *header, const char *name) {
    if (!header || !name) return NULL;
    for (size_t index = 0; index < header->tensor_count; index++) {
        if (!strcmp(header->tensors[index].name, name)) return &header->tensors[index];
    }
    return NULL;
}

uint64_t h3_st_tensor_elements(const h3_st_tensor *tensor) {
    if (!tensor) return 0;
    uint64_t elements = 1;
    for (int dimension = 0; dimension < tensor->ndim; dimension++) {
        if (tensor->shape[dimension] != 0 &&
            elements > UINT64_MAX / tensor->shape[dimension]) return 0;
        elements *= tensor->shape[dimension];
    }
    return elements;
}

uint64_t h3_st_tensor_nbytes(const h3_st_tensor *tensor) {
    if (!tensor) return 0;
    return tensor->data_end - tensor->data_begin;
}

void h3_st_quant_info_init(h3_st_quant_info *info) {
    if (!info) return;
    memset(info, 0, sizeof(*info));
    info->group_size = -1;
    info->convrot = -1;
    info->convrot_groupsize = -1;
}

static const char *h3_json_find_key(const char *json, const char *key) {
    if (!json || !key) return NULL;
    size_t key_len = strlen(key);
    const char *at = json;
    while ((at = strstr(at, key)) != NULL) {
        int quoted = 0;
        if (at > json && at[-1] == '"' && at[key_len] == '"') {
            quoted = 1;
        } else if (at > json + 1 && at[-2] == '\\' && at[-1] == '"' &&
                   at[key_len] == '\\' && at[key_len + 1] == '"') {
            /* Nested JSON stored as an escaped string value. */
            quoted = 2;
        }
        if (quoted) {
            const char *after = at + key_len + (quoted == 2 ? 2 : 1);
            while (*after == ' ' || *after == '\t' || *after == '\n' ||
                   *after == '\r') after++;
            if (*after == ':') return after + 1;
        }
        at += key_len;
    }
    return NULL;
}

static int h3_json_read_string_value(const char *at, char *out, size_t out_size) {
    if (!at || !out || !out_size) return 0;
    while (*at == ' ' || *at == '\t' || *at == '\n' || *at == '\r') at++;
    int escaped_quote = 0;
    if (at[0] == '\\' && at[1] == '"') {
        escaped_quote = 1;
        at += 2;
    } else if (*at == '"') {
        at++;
    } else {
        return 0;
    }
    size_t length = 0;
    while (*at) {
        if (escaped_quote) {
            if (at[0] == '\\' && at[1] == '"') break;
        } else if (*at == '"') {
            break;
        }
        if (*at == '\\' && at[1]) at++;
        if (length + 1 >= out_size) return 0;
        out[length++] = *at++;
    }
    if (!*at) return 0;
    out[length] = '\0';
    return 1;
}

static int h3_json_read_int_value(const char *at, int *out) {
    if (!at || !out) return 0;
    while (*at == ' ' || *at == '\t' || *at == '\n' || *at == '\r') at++;
    if (*at < '0' || *at > '9') {
        if (*at == 't' && !strncmp(at, "true", 4)) {
            *out = 1;
            return 1;
        }
        if (*at == 'f' && !strncmp(at, "false", 5)) {
            *out = 0;
            return 1;
        }
        return 0;
    }
    long value = 0;
    while (*at >= '0' && *at <= '9') {
        value = value * 10 + (*at - '0');
        if (value > INT_MAX) return 0;
        at++;
    }
    *out = (int)value;
    return 1;
}

static void h3_st_quant_scan_blob(const char *blob, h3_st_quant_info *info) {
    if (!blob || !info) return;
    const char *format_at = h3_json_find_key(blob, "format");
    if (!format_at) format_at = h3_json_find_key(blob, "quant_format");
    if (format_at && !info->format[0]) {
        h3_json_read_string_value(format_at, info->format, sizeof(info->format));
    }
    if (!info->format[0]) {
        if (strstr(blob, "asym_w4a8_int8")) {
            snprintf(info->format, sizeof(info->format), "asym_w4a8_int8");
        } else if (strstr(blob, "nvfp4")) {
            snprintf(info->format, sizeof(info->format), "nvfp4_awq_like");
        }
    }
    const char *group_at = h3_json_find_key(blob, "group_size");
    if (group_at && info->group_size < 0) {
        h3_json_read_int_value(group_at, &info->group_size);
    }
    const char *convrot_at = h3_json_find_key(blob, "convrot");
    if (convrot_at && info->convrot < 0) {
        h3_json_read_int_value(convrot_at, &info->convrot);
    }
    const char *cgs_at = h3_json_find_key(blob, "convrot_groupsize");
    if (cgs_at && info->convrot_groupsize < 0) {
        h3_json_read_int_value(cgs_at, &info->convrot_groupsize);
    }
}

int h3_st_quant_info_from_header(const h3_st_header *header,
                                 h3_st_quant_info *info) {
    if (!info) return 0;
    h3_st_quant_info_init(info);
    if (!header) return 0;
    if (header->metadata_json && header->metadata_json[0]) {
        info->has_metadata = 1;
        h3_st_quant_scan_blob(header->metadata_json, info);
    }
    /* Tensor-name hints when metadata is missing or incomplete. */
    int has_codebook = 0;
    int has_s_rel = 0;
    int has_s_channel = 0;
    int has_weight_scale = 0;
    int has_u8_weight = 0;
    for (size_t index = 0; index < header->tensor_count; index++) {
        const char *name = header->tensors[index].name;
        if (!name) continue;
        if (strstr(name, "weight_codebook")) has_codebook = 1;
        if (strstr(name, "weight_s_rel")) has_s_rel = 1;
        if (strstr(name, "weight_s_channel")) has_s_channel = 1;
        if (strstr(name, "weight_scale")) has_weight_scale = 1;
        if (header->tensors[index].dtype == H3_DTYPE_U8 &&
            strstr(name, ".weight") && !strstr(name, "weight_")) {
            has_u8_weight = 1;
        }
    }
    if (!info->format[0]) {
        if (has_codebook && has_s_rel && has_s_channel) {
            snprintf(info->format, sizeof(info->format), "asym_w4a8_int8");
        } else if (has_u8_weight && has_weight_scale) {
            snprintf(info->format, sizeof(info->format), "nvfp4_awq_like");
        } else if (has_weight_scale) {
            snprintf(info->format, sizeof(info->format), "int8_scale_convrot_like");
        }
    }
    return info->format[0] != '\0' || info->has_metadata;
}

int h3_st_read_data(const h3_st_header *header, const h3_st_tensor *tensor,
                    void *data, size_t bytes, char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!header || !header->path || !tensor || (!data && bytes != 0)) {
        if (error && error_size) snprintf(error, error_size, "invalid tensor read request");
        return 0;
    }
    uint64_t tensor_bytes = tensor->data_end - tensor->data_begin;
    if (tensor_bytes != bytes) {
        if (error && error_size) {
            snprintf(error, error_size, "%s: expected %llu bytes, got %zu",
                     tensor->name ? tensor->name : "tensor",
                     (unsigned long long)tensor_bytes, bytes);
        }
        return 0;
    }
    int descriptor = open(header->path, O_RDONLY);
    if (descriptor < 0) {
        if (error && error_size) {
            snprintf(error, error_size, "%s: %s", header->path, strerror(errno));
        }
        return 0;
    }
    size_t done = 0;
    while (done < bytes) {
        size_t remaining = bytes - done;
        size_t chunk = remaining < (size_t)(1u << 30) ? remaining :
                                                            (size_t)(1u << 30);
        ssize_t count = pread(descriptor, (unsigned char *)data + done, chunk,
                              (off_t)(tensor->file_offset + done));
        if (count <= 0) {
            if (error && error_size) {
                snprintf(error, error_size, "%s: cannot read %s: %s", header->path,
                         tensor->name ? tensor->name : "tensor",
                         count < 0 ? strerror(errno) : "unexpected end of file");
            }
            close(descriptor);
            return 0;
        }
        done += (size_t)count;
    }
    close(descriptor);
    return 1;
}

static int h3_has_suffix(const char *value, const char *suffix) {
    size_t length = strlen(value);
    size_t suffix_length = strlen(suffix);
    return length >= suffix_length && !strcmp(value + length - suffix_length, suffix);
}

int h3_st_inventory_dir(const char *directory, h3_component_info *info,
                        char *error, size_t error_size) {
    if (!directory || !info) return 0;
    memset(info, 0, sizeof(*info));
    DIR *stream = opendir(directory);
    if (!stream) {
        if (error && error_size) snprintf(error, error_size, "%s: %s", directory, strerror(errno));
        return 0;
    }
    struct dirent *entry;
    while ((entry = readdir(stream)) != NULL) {
        if (entry->d_name[0] == '.' || !h3_has_suffix(entry->d_name, ".safetensors")) continue;
        size_t length = strlen(directory) + strlen(entry->d_name) + 2;
        char *path = malloc(length);
        if (!path) {
            if (error && error_size) snprintf(error, error_size, "out of memory");
            closedir(stream);
            return 0;
        }
        snprintf(path, length, "%s/%s", directory, entry->d_name);
        h3_st_header header;
        int ok = h3_st_read_header(path, &header, error, error_size);
        free(path);
        if (!ok) {
            closedir(stream);
            return 0;
        }
        info->files++;
        info->bytes += header.file_size;
        info->tensors += header.tensor_count;
        for (size_t index = 0; index < header.tensor_count; index++) {
            info->tensor_bytes += header.tensors[index].data_end -
                                  header.tensors[index].data_begin;
        }
        h3_st_free_header(&header);
    }
    closedir(stream);
    if (!info->files) {
        if (error && error_size) snprintf(error, error_size, "%s: no safetensors files", directory);
        return 0;
    }
    return 1;
}

static float h3_st_bf16_bits_to_f32(uint16_t bits) {
    union {
        uint32_t u;
        float f;
    } v;
    v.u = ((uint32_t)bits) << 16;
    return v.f;
}

static float h3_st_f16_bits_to_f32(uint16_t bits) {
    const uint32_t sign = (uint32_t)(bits & 0x8000u) << 16;
    const uint32_t exp = (bits >> 10) & 0x1fu;
    const uint32_t mant = bits & 0x3ffu;
    union {
        uint32_t u;
        float f;
    } v;
    if (exp == 0) {
        if (mant == 0) {
            v.u = sign;
            return v.f;
        }
        /* subnormal — match h3_w4a8_load_f32 */
        float m = (float)mant / 1024.0f;
        float val = ldexpf(m, -14);
        v.f = val;
        if (sign) v.f = -v.f;
        return v.f;
    }
    if (exp == 31) {
        v.u = sign | 0x7f800000u | (mant << 13);
        return v.f;
    }
    v.u = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
    return v.f;
}

float *h3_st_load_f32(const h3_st_header *header, const char *name,
                      size_t *out_elements, h3_dtype *out_dtype, char *error,
                      size_t error_size) {
    const h3_st_tensor *tensor;
    size_t elements;
    float *out = NULL;
    size_t i;

    if (out_elements) *out_elements = 0;
    if (out_dtype) *out_dtype = H3_DTYPE_UNKNOWN;
    if (!header || !name) {
        if (error && error_size)
            snprintf(error, error_size, "invalid h3_st_load_f32 args");
        return NULL;
    }
    tensor = h3_st_find(header, name);
    if (!tensor) {
        if (error && error_size)
            snprintf(error, error_size, "missing tensor: %s", name);
        return NULL;
    }
    if (tensor->dtype != H3_DTYPE_F32 && tensor->dtype != H3_DTYPE_BF16 &&
        tensor->dtype != H3_DTYPE_F16) {
        if (error && error_size)
            snprintf(error, error_size, "%s dtype %s not convertible to F32",
                     name, h3_dtype_name(tensor->dtype));
        return NULL;
    }
    elements = (size_t)h3_st_tensor_elements(tensor);
    if (elements == 0) {
        if (error && error_size)
            snprintf(error, error_size, "%s is empty", name);
        return NULL;
    }
    out = (float *)malloc(elements * sizeof(float));
    if (!out) {
        if (error && error_size)
            snprintf(error, error_size, "out of memory loading %s", name);
        return NULL;
    }
    if (tensor->dtype == H3_DTYPE_F32) {
        if (!h3_st_read_data(header, tensor, out, elements * sizeof(float),
                             error, error_size)) {
            free(out);
            return NULL;
        }
    } else {
        uint16_t *raw = (uint16_t *)malloc(elements * sizeof(uint16_t));
        if (!raw) {
            if (error && error_size)
                snprintf(error, error_size, "out of memory reading %s", name);
            free(out);
            return NULL;
        }
        if (!h3_st_read_data(header, tensor, raw, elements * sizeof(uint16_t),
                             error, error_size)) {
            free(raw);
            free(out);
            return NULL;
        }
        if (tensor->dtype == H3_DTYPE_BF16) {
            for (i = 0; i < elements; i++)
                out[i] = h3_st_bf16_bits_to_f32(raw[i]);
        } else {
            for (i = 0; i < elements; i++)
                out[i] = h3_st_f16_bits_to_f32(raw[i]);
        }
        free(raw);
    }
    if (out_elements) *out_elements = elements;
    if (out_dtype) *out_dtype = tensor->dtype;
    return out;
}

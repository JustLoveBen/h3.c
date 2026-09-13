#include "h3_lora.h"
#include <stdio.h>
#include <string.h>
static int fails;
static void expect(int c, const char *m) {
    if (!c) {
        fprintf(stderr, "FAIL %s\n", m);
        fails++;
    }
}
int main(void) {
    unsigned b;
    const char *suf;
    int is_a;
    char err[128];
    expect(h3_lora_parse_name("blocks.3.attn.qkv_proj.lora_A.weight", &b, &suf,
                              &is_a, err, sizeof err) == 1,
           "peft qkv A");
    expect(b == 3 && strcmp(suf, "qkv") == 0 && is_a == 1, "qkv A fields");
    expect(h3_lora_parse_name("blocks.3.attn.qkv_proj.lora_B.weight", &b, &suf,
                              &is_a, err, sizeof err) == 1,
           "peft qkv B");
    expect(is_a == 0, "B");
    expect(h3_lora_name_is_vsa("blocks.0.attn.to_gate_compress.weight") == 1,
           "vsa");
    expect(h3_lora_parse_name("token_refiner.blocks.0.attn.qkv_proj.lora_A.weight",
                              &b, &suf, &is_a, err, sizeof err) == 0,
           "token_refiner not dit block");
    expect(h3_lora_load("/no/such.safetensors", 1.0f, NULL, err, sizeof err) == 0,
           "missing file");
    expect(strstr(err, "非金线") != NULL || strstr(err, "cannot") != NULL,
           "missing file err");
    /* apply: r=1, M=2, K=2, N=2, A=[1,1], B=[1,1] → delta all-ones * scale */
    {
        h3_lora_pair p;
        float A[2] = {1.f, 1.f}, B[2] = {1.f, 1.f};
        float x[4] = {1.f, 0.f, 0.f, 1.f};
        float y[4] = {0, 0, 0, 0};
        p.A = A;
        p.B = B;
        p.r = 1;
        p.k = 2;
        p.n = 2;
        expect(h3_lora_apply_f32(&p, 2.0f, x, y, 2, err, sizeof err) == 1,
               "apply");
        expect(y[0] == 2.f && y[1] == 2.f, "row0");
    }
    return fails ? 1 : (printf("test_h3_lora: PASS\n"), 0);
}

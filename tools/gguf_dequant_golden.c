/* Golden-vector harness: runs llama.cpp's own Q8_0 / IQ2_XS dequantizers on
 * block bytes from a file and writes fp32 results. Used to prove
 * loader/gguf_dequant.py bit-exact against upstream llama.cpp.
 *
 * Build (from the qwen-sm120-runtime repo root), assuming a llama.cpp build:
 *   gcc -O2 -DGGML_COMMON_DECL_C -I <llama.cpp>/ggml/src tools/gguf_dequant_golden.c \
 *       -L <llama.cpp>/build-sm120/bin -lggml-base \
 *       -Wl,-rpath,<llama.cpp>/build-sm120/bin -o /tmp/gguf_dequant_golden
 * Usage: gguf_dequant_golden q8_0|iq2_xs blocks.bin out.f32
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ggml-common.h"

extern void dequantize_row_q8_0(const block_q8_0 *x, float *y, int64_t k);
extern void dequantize_row_iq2_xs(const block_iq2_xs *x, float *y, int64_t k);

int main(int argc, char **argv) {
    if (argc != 4) {
        fprintf(stderr, "usage: %s q8_0|iq2_xs blocks.bin out.f32\n", argv[0]);
        return 2;
    }
    int is_iq2 = strcmp(argv[1], "iq2_xs") == 0;
    int is_q8 = strcmp(argv[1], "q8_0") == 0;
    if (!is_iq2 && !is_q8) {
        fprintf(stderr, "unknown type %s\n", argv[1]);
        return 2;
    }
    FILE *fin = fopen(argv[2], "rb");
    if (!fin) { perror("open input"); return 1; }
    if (fseek(fin, 0, SEEK_END) != 0) { return 1; }
    long size = ftell(fin);
    if (fseek(fin, 0, SEEK_SET) != 0) { return 1; }
    const size_t block_bytes = is_iq2 ? sizeof(block_iq2_xs) : sizeof(block_q8_0);
    const int64_t block_elems = is_iq2 ? QK_K : 32;
    if (size <= 0 || (size_t)size % block_bytes != 0) {
        fprintf(stderr, "input size %ld not a multiple of %zu\n", size, block_bytes);
        return 1;
    }
    const int64_t nblocks = (int64_t)((size_t)size / block_bytes);
    char *x = malloc((size_t)size);
    float *y = malloc((size_t)nblocks * (size_t)block_elems * sizeof(float));
    if (!x || !y) { return 1; }
    if (fread(x, 1, (size_t)size, fin) != (size_t)size) { return 1; }
    fclose(fin);
    if (is_iq2) {
        dequantize_row_iq2_xs((const block_iq2_xs *)x, y, nblocks * block_elems);
    } else {
        dequantize_row_q8_0((const block_q8_0 *)x, y, nblocks * block_elems);
    }
    FILE *fout = fopen(argv[3], "wb");
    if (!fout) { perror("open output"); return 1; }
    if (fwrite(y, sizeof(float), (size_t)nblocks * (size_t)block_elems, fout)
        != (size_t)nblocks * (size_t)block_elems) { return 1; }
    fclose(fout);
    free(x);
    free(y);
    return 0;
}

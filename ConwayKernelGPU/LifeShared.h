#pragma once
#include <stdint.h>
#include <simd/simd.h>

#ifdef __cplusplus
extern "C" {
#endif

enum { LIFE_NMAX = 16 };

typedef struct {
    uint32_t W;
    uint32_t H;
    uint32_t strideWords;
    uint32_t N;
    uint32_t subtractCenter;
    uint16_t kernelRow[LIFE_NMAX];
    uint32_t _pad;
} SimParams;

typedef struct {
    uint8_t next[2][257];
} RuleLUT;

typedef struct {
    uint32_t W;
    uint32_t H;
    uint32_t strideWords;

    simd_uint2 viewSizePx;
    simd_float2 centerCell;
    float zoom;
    uint32_t _pad0;

    uint32_t aliveARGB;
    uint32_t deadARGB;
} ViewParams;

#ifdef __cplusplus
}
#endif

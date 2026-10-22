/* nd_simd.h - the int8 vector kernels behind nd_quant's batched matmuls.
 *
 * In a header so that esp32/nd_bench times exactly the code the engine runs.
 * On the ESP32-S3 and ESP32-P4 these are PIE inline assembly; elsewhere a C
 * loop computes the same integers, so the host build checks what the board
 * does. Include after sdkconfig.h on the ESP32: the target selects the code.
 */
#ifndef ND_SIMD_H
#define ND_SIMD_H

#include <stdint.h>

static inline int32_t nd_dot128_ref(const int8_t *a, const int8_t *b)
{
    int32_t  r = 0;
    uint32_t i;
    for (i = 0; i < 128; i++)
        r += (int32_t)a[i] * b[i];
    return r;
}

/* sum(a[i] * b[i]) over 128 int8s; both 16-byte aligned. */
static inline int32_t nd_dot128_s8(const int8_t *a, const int8_t *b)
{
#if defined(ESP_PLATFORM) && defined(CONFIG_IDF_TARGET_ESP32S3)
    int32_t r;
    __asm__ volatile(
        "ee.zero.accx\n"
        "ee.vld.128.ip q0, %1, 16\n"  "ee.vld.128.ip q1, %2, 16\n"
        "ee.vld.128.ip q2, %1, 16\n"  "ee.vld.128.ip q3, %2, 16\n"
        "ee.vmulas.s8.accx q0, q1\n"
        "ee.vld.128.ip q4, %1, 16\n"  "ee.vld.128.ip q5, %2, 16\n"
        "ee.vmulas.s8.accx q2, q3\n"
        "ee.vld.128.ip q6, %1, 16\n"  "ee.vld.128.ip q7, %2, 16\n"
        "ee.vmulas.s8.accx q4, q5\n"
        "ee.vld.128.ip q0, %1, 16\n"  "ee.vld.128.ip q1, %2, 16\n"
        "ee.vmulas.s8.accx q6, q7\n"
        "ee.vld.128.ip q2, %1, 16\n"  "ee.vld.128.ip q3, %2, 16\n"
        "ee.vmulas.s8.accx q0, q1\n"
        "ee.vld.128.ip q4, %1, 16\n"  "ee.vld.128.ip q5, %2, 16\n"
        "ee.vmulas.s8.accx q2, q3\n"
        "ee.vld.128.ip q6, %1, 16\n"  "ee.vld.128.ip q7, %2, 16\n"
        "ee.vmulas.s8.accx q4, q5\n"
        "ee.vmulas.s8.accx q6, q7\n"
        "rur.accx_0 %0\n"           /* |sum| <= 128*127*127: fits 32 bits */
        : "=r"(r), "+r"(a), "+r"(b) :: "memory");
    return r;
#elif defined(ESP_PLATFORM) && defined(CONFIG_IDF_TARGET_ESP32P4)
    /* The same kernel in the P4's RISC-V PIE. Revisions < 3 (xesppie) and
     * >= 3 (xespv) encode these differently; -march picks the encoding.
     * PIE address and result operands must be x8-x15: once inlined, a plain
     * "r" operand let GCC pick a7 or t2, which the assembler rejects. */
    register int32_t       r  __asm__("a3");
    register const int8_t *pa __asm__("a4") = a;
    register const int8_t *pb __asm__("a5") = b;
    __asm__ volatile(
        "esp.zero.xacc\n"
        "esp.vld.128.ip q0, %1, 16\n"  "esp.vld.128.ip q1, %2, 16\n"
        "esp.vld.128.ip q2, %1, 16\n"  "esp.vld.128.ip q3, %2, 16\n"
        "esp.vmulas.s8.xacc q0, q1\n"
        "esp.vld.128.ip q4, %1, 16\n"  "esp.vld.128.ip q5, %2, 16\n"
        "esp.vmulas.s8.xacc q2, q3\n"
        "esp.vld.128.ip q6, %1, 16\n"  "esp.vld.128.ip q7, %2, 16\n"
        "esp.vmulas.s8.xacc q4, q5\n"
        "esp.vld.128.ip q0, %1, 16\n"  "esp.vld.128.ip q1, %2, 16\n"
        "esp.vmulas.s8.xacc q6, q7\n"
        "esp.vld.128.ip q2, %1, 16\n"  "esp.vld.128.ip q3, %2, 16\n"
        "esp.vmulas.s8.xacc q0, q1\n"
        "esp.vld.128.ip q4, %1, 16\n"  "esp.vld.128.ip q5, %2, 16\n"
        "esp.vmulas.s8.xacc q2, q3\n"
        "esp.vld.128.ip q6, %1, 16\n"  "esp.vld.128.ip q7, %2, 16\n"
        "esp.vmulas.s8.xacc q4, q5\n"
        "esp.vmulas.s8.xacc q6, q7\n"
        "esp.movx.r.xacc.l %0\n"     /* |sum| <= 128*127*127: fits 32 bits */
        : "=r"(r), "+r"(pa), "+r"(pb) :: "memory");
    return r;
#else
    return nd_dot128_ref(a, b);
#endif
}

#endif /* ND_SIMD_H */

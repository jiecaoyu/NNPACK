#pragma once

#include <stdbool.h>

#include <nnpack/arm_neon.h>
#include <nnpack/macros.h>

#include <stdio.h>

static NNP_INLINE void winograd_f4k3_input_transform(
	const float32x4_t d0,
	const float32x4_t d1,
	const float32x4_t d2,
	const float32x4_t d3,
	const float32x4_t d4,
	const float32x4_t d5,
	float32x4_t transform0[restrict static 1],
	float32x4_t transform1[restrict static 1],
	float32x4_t transform2[restrict static 1],
	float32x4_t transform3[restrict static 1],
	float32x4_t transform4[restrict static 1],
	float32x4_t transform5[restrict static 1])
{
    /*
     * w0 = - (-4.0 * d0 + d2) + (-4.0 * d2 + d4)
     * w1 = + (-4.0 * d1 + d3) + (-4.0 * d2 + d4)
     * w2 = - (-4.0 * d1 + d3) + (-4.0 * d2 + d4)
     * w3 = - 2.0 * (d1 - d3) + (-1.0) * (d2 - d4)
     * w4 = + 2.0 * (d1 - d3) + (-1.0) * (d2 - d4)
     * w5 = - (-4.0 * d1 + d3) + (-4.0 * d3 + d5)
     */
    const float32x4_t _n4d0_pls_d2 = vmlaq_n_f32(d2, d0, -4.0f);
    const float32x4_t _n4d2_pls_d4 = vmlaq_n_f32(d4, d2, -4.0f);
    const float32x4_t _n4d1_pls_d3 = vmlaq_n_f32(d3, d1, -4.0f);
    const float32x4_t _n4d3_pls_d5 = vmlaq_n_f32(d5, d3, -4.0f);

    const float32x4_t _2_d1_sub_d3 = vdupq_n_f32(2.0f) * vsubq_f32(d1, d3);
    const float32x4_t _n1_d2_sub_d4 = vsubq_f32(d4, d2);

    *transform0 = vsubq_f32(_n4d2_pls_d4, _n4d0_pls_d2);
    *transform1 = vaddq_f32(_n4d2_pls_d4, _n4d1_pls_d3);
    *transform2 = vsubq_f32(_n4d2_pls_d4, _n4d1_pls_d3);
    *transform3 = vsubq_f32(_n1_d2_sub_d4, _2_d1_sub_d3);
    *transform4 = vaddq_f32(_n1_d2_sub_d4, _2_d1_sub_d3);
    *transform5 = vsubq_f32(_n4d3_pls_d5, _n4d1_pls_d3);
    return;
}

static NNP_INLINE void winograd_f4k3_kernel_transform(
	const float32x4_t g0, const float32x4_t g1, const float32x4_t g2,
	float32x4_t transform0[restrict static 1],
	float32x4_t transform1[restrict static 1],
	float32x4_t transform2[restrict static 1],
	float32x4_t transform3[restrict static 1],
	float32x4_t transform4[restrict static 1],
        float32x4_t transform5[restrict static 1],
        bool rescale_coefficients)
{
    /*
     * w0 = g0 * (1.0 / 4)
     * w1 = ((g0 + g2) + g1) * (-1.0 / 6)
     * w2 = ((g0 + g2) - g1) * (-1.0 / 6)
     * w3 = ((g0 + 4 * g2) + 2 * g1) * (1.0 / 24)
     * w4 = ((g0 + 4 * g2) - 2 * g1) * (1.0 / 24)
     * w5 = g2
     */


    float32x4_t w0 = g0;
    /*
     * Compute
     *   w2 := g0 + g2
     *   w4 := g0 + 4 * g2
     */
    const float32x4_t const_4 = vdupq_n_f32(4.0f);
    float32x4_t w2 = g0 + g2;
    float32x4_t w4 = vmuladdq_f32(g0, const_4, g2);

    /*
     * Compute
     *   w1 = (g0 + g2) + g1
     *   w2 = (g0 + g2) - g1
     *   w3 = (g0 + 4 * g2) + 2 * g1
     */
    const float32x4_t two_g1 = g1 * vdupq_n_f32(2.0f);
    float32x4_t w1 = w2 + g1;
    w2 = w2 - g1;
    float32x4_t w3 = w4 + two_g1;
    w4 = w4 - two_g1;

    if (rescale_coefficients) {
        const float32x4_t rcp_4 = vdupq_n_f32(0x1.000000p-2f);
        w0 *= rcp_4;

        const float32x4_t rcp_6 = vdupq_n_f32(-0x1.555556p-3f);
        w1 *= rcp_6;
        w2 *= rcp_6;

        const float32x4_t rcp_24 = vdupq_n_f32(0x1.555555p-5f);
        w3 *= rcp_24;
        w4 *= rcp_24;
    }

    *transform0 = w0;
    *transform1 = w1;
    *transform2 = w2;
    *transform3 = w3;
    *transform4 = w4;
    *transform5 = g2;
    return;
}

static NNP_INLINE void winograd_f4k3_output_transformq(
	const float32x4_t m0,
	const float32x4_t m1,
	const float32x4_t m2,
	const float32x4_t m3,
	const float32x4_t m4,
	const float32x4_t m5,
	float32x4_t output0[restrict static 1],
	float32x4_t output1[restrict static 1],
	float32x4_t output2[restrict static 1],
	float32x4_t output3[restrict static 1])
{
    /*
     * output0 = m0 + (m1 + m2) + (m3 + m4)
     * output1 = (m1 - m2) + 2.0 * (m3 - m4)
     * output2 = (m1 + m2) + 4.0 * (m3 + m4)
     * output3 = (m1 - m2) + 8.0 * (m3 - m4) + m5;
     */
    const float32x4_t m1_pls_m2 = vaddq_f32(m1, m2);
    const float32x4_t m1_sub_m2 = vsubq_f32(m1, m2);
    const float32x4_t m3_pls_m4 = vaddq_f32(m3, m4);
    const float32x4_t m3_sub_m4 = vsubq_f32(m3, m4);

    *output0 = vaddq_f32(m0, vaddq_f32(m1_pls_m2, m3_pls_m4));
    *output1 = vmlaq_n_f32(m1_sub_m2, m3_sub_m4, 2.0f);
    *output2 = vmlaq_n_f32(m1_pls_m2, m3_pls_m4, 4.0f);
    *output3 = vaddq_f32(vmlaq_n_f32(m1_sub_m2, m3_sub_m4, 8.0f), m5);
    return;
}

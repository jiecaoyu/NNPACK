#include <stdint.h>
#include <stdio.h>
#include <stddef.h>
#include <stdlib.h>

#include <nnpack/arm_neon.h>
#include <nnpack/activations.h>
#include <nnpack/macros.h>
#include <nnpack/utils.h>

#include <neon/winograd/f4x4k3x3.h>
#include <neon/transpose.h>

void nnp_iwt6x6_3x3_with_offset__neon(
	const float data[restrict static 1],
	void* transform,
	size_t data_stride,
	size_t transform_stride,
	uint32_t row_count,
	uint32_t column_count,
	uint32_t row_offset,
	uint32_t column_offset)
{
    NNP_SIMD_ALIGN float32x4_t wd[16];
    NNP_SIMD_ALIGN float block[8][8];
    {
        const float32x4_t vzero = vmovq_n_f32(0.0f);
        for (float *block_ptr = &block[0][0], *block_end = &block[8][8];
                block_ptr != block_end; block_ptr += 4) {
            vst1q_f32(block_ptr, vzero);
        }
    }
    for (size_t i = 0; i < row_count; i++) {
        for (size_t j = 0; j < column_count; j++) {
            block[row_offset + i][column_offset + j] = data[i * data_stride + j];
        }
    }
    for (size_t col = 0; col < 2; col++) {
        winograd_f4k3_input_transform(
                vld1q_f32(&block[0][col * 4]),
                vld1q_f32(&block[1][col * 4]),
                vld1q_f32(&block[2][col * 4]),
                vld1q_f32(&block[3][col * 4]),
                vld1q_f32(&block[4][col * 4]),
                vld1q_f32(&block[5][col * 4]),
                &wd[0 + col * 4], &wd[1 + col * 4], &wd[ 2 + col * 4], &wd[3 + col * 4],
                &wd[8 + col * 4], &wd[9 + col * 4]);
        wd[10 + col * 4] = vld1q_f32(&block[6][col * 4]);
        wd[11 + col * 4] = vld1q_f32(&block[7][col * 4]);
    }

    for (size_t col = 0; col < 2; ++col) {
        float32x4_t vout0,vout1,vout2,vout3,vout4,vout5,vout6,vout7;
	float32x4x4_t vin0123 = vld4q_f32((const float*) &wd[0 + col * 8]);
	float32x4x4_t vin4567 = vld4q_f32((const float*) &wd[4 + col * 8]);
	winograd_f4k3_input_transform(
                vin0123.val[0], vin0123.val[1], vin0123.val[2], vin0123.val[3],
                vin4567.val[0], vin4567.val[1],
                &vout0, &vout1, &vout2, &vout3, &vout4, &vout5);
        if (col == 0) {
            vst1q_f32(transform, vout0);
            transform += transform_stride;
            vst1q_f32(transform, vout1);
            transform += transform_stride;
            vst1q_f32(transform, vout2);
            transform += transform_stride;
            vst1q_f32(transform, vout3);
            transform += transform_stride;
            vst1q_f32(transform, vout4);
            transform += transform_stride;
            vst1q_f32(transform, vout5);
            transform += transform_stride;
        }
        else {
            vst1q_f32(transform, vcombine_f32(vget_low_f32(vout0), vget_low_f32(vout1)));
            transform += transform_stride;
            vst1q_f32(transform, vcombine_f32(vget_low_f32(vout2), vget_low_f32(vout3)));
            transform += transform_stride;
            vst1q_f32(transform, vcombine_f32(vget_low_f32(vout4), vget_low_f32(vout5)));
        }
    }

    return;
}

void nnp_kwt6x6_3x3__neon(
        const float g[restrict static 9],
        float transform[restrict static 1],
        size_t stride_g,
        size_t transform_stride,
        uint32_t row_count,
        uint32_t column_count,
        uint32_t row_offset,
        uint32_t column_offset)
{
    transform_stride /= sizeof(float);
    const float32x4_t g0 = vld1q_f32(g);
    const float32x4_t g1 = vld1q_f32(g + 3);
    // g2[3] is junk
    const float32x4_t g2 = vextq_f32(vld1q_f32(g + 5), vld1q_f32(g + 5), 1);
    NNP_SIMD_ALIGN float32x4_t w[8];
    winograd_f4k3_kernel_transform(g0, g1, g2,
            &w[0], &w[1], &w[2], &w[3], &w[4], &w[5],
            true /* rescale coefficients */);
    neon_transpose4x4_inplace_f32(&w[0], &w[1], &w[2], &w[3]);
    neon_transpose4x4_inplace_f32(&w[4], &w[5], &w[6], &w[7]);
    NNP_SIMD_ALIGN float32x4_t wg[8][2];
    winograd_f4k3_kernel_transform(w[0], w[1], w[2],
            &wg[0][0], &wg[1][0], &wg[2][0], &wg[3][0],
            &wg[4][0], &wg[5][0],
            true /* rescale coefficients */);
    winograd_f4k3_kernel_transform(w[4], w[5], w[6],
            &wg[0][1], &wg[1][1], &wg[2][1], &wg[3][1],
            &wg[4][1], &wg[5][1],
            true /* rescale coefficients */);
    vst1q_f32(transform, wg[0][0]);
    transform += transform_stride;
    vst1q_f32(transform, wg[1][0]);
    transform += transform_stride;
    vst1q_f32(transform, wg[2][0]);
    transform += transform_stride;
    vst1q_f32(transform, wg[3][0]);
    transform += transform_stride;
    vst1q_f32(transform, wg[4][0]);
    transform += transform_stride;
    vst1q_f32(transform, wg[5][0]);
    transform += transform_stride;
    vst1q_f32(transform, vcombine_f32(vget_low_f32(wg[0][1]), vget_low_f32(wg[1][1])));
    transform += transform_stride;
    vst1q_f32(transform, vcombine_f32(vget_low_f32(wg[2][1]), vget_low_f32(wg[3][1])));
    transform += transform_stride;
    vst1q_f32(transform, vcombine_f32(vget_low_f32(wg[4][1]), vget_low_f32(wg[5][1])));
    return;
}

#if !NNP_INFERENCE_ONLY
void nnp_kwt6x6_3Rx3R__neon(
        const float g[restrict static 9],
        float transform[restrict static 1],
        size_t stride_g,
        size_t transform_stride,
        uint32_t row_count,
        uint32_t column_count,
        uint32_t row_offset,
        uint32_t column_offset)
{
    fprintf(stderr, "Error: unsupported kernel nnp_kwt6x6_3Rx3R__neon\n");
    exit(1);
    return;
}

void nnp_owt6x6_3x3__neon(
        const void *restrict transform,
        float output[restrict static 1],
        size_t transform_stride,
        size_t output_stride,
        uint32_t row_count,
        uint32_t column_count,
        uint32_t row_offset,
        uint32_t column_offset)
{
    fprintf(stderr, "Error: unsupported kernel nnp_owt6x6_3x3__neon\n");
    exit(1);
    return;
}
#endif /* !NNP_INFERENCE_ONLY */

void nnp_owt6x6_3x3_with_bias__neon(
        const void *restrict transform,
        float output[restrict static 1],
        const float bias[restrict static 1],
        size_t transform_stride,
        size_t output_stride,
        uint32_t row_count,
        uint32_t column_count)
{
    // main part
    NNP_SIMD_ALIGN float32x4_t buffer[8];
    const float32x4_t m0 = vld1q_f32(transform); transform += transform_stride;
    const float32x4_t m1 = vld1q_f32(transform); transform += transform_stride;
    const float32x4_t m2 = vld1q_f32(transform); transform += transform_stride;
    const float32x4_t m3 = vld1q_f32(transform); transform += transform_stride;
    const float32x4_t m4 = vld1q_f32(transform); transform += transform_stride;
    const float32x4_t m5 = vld1q_f32(transform); transform += transform_stride;
    winograd_f4k3_output_transformq(
            m0, m1, m2, m3, m4, m5,
            &buffer[0], &buffer[1], &buffer[2], &buffer[3]);

    // rest part
    const size_t transform_stride_rest = transform_stride - 2 * sizeof(float32_t);
    const float32x4_t m6  = vld1q_f32(transform); transform += 2 * sizeof(float32_t);
    const float32x4_t m7  = vld1q_f32(transform); transform += transform_stride_rest;
    const float32x4_t m8  = vld1q_f32(transform); transform += 2 * sizeof(float32_t);
    const float32x4_t m9 = vld1q_f32(transform); transform += transform_stride_rest;
    const float32x4_t m10 = vld1q_f32(transform); transform += 2 * sizeof(float32_t);
    const float32x4_t m11 = vcombine_f32(vld1_f32(transform), vmov_n_f32(0.0f));
    winograd_f4k3_output_transformq(
            m6, m7, m8, m9, m10, m11,
            &buffer[4], &buffer[5], &buffer[6], &buffer[7]);
    // final
    neon_transpose4x4_inplace_f32(&buffer[0], &buffer[1], &buffer[2], &buffer[3]);
    neon_transpose4x4_inplace_f32(&buffer[4], &buffer[5], &buffer[6], &buffer[7]);

    NNP_SIMD_ALIGN float32x4_t block[4];
    winograd_f4k3_output_transformq(
            buffer[0], buffer[1], buffer[2], buffer[3], buffer[4], buffer[5],
            &block[0], &block[1], &block[2], &block[3]);

    const float32x4_t vbias = vdupq_n_f32(*bias);
    block[0] = vaddq_f32(block[0], vbias);
    block[1] = vaddq_f32(block[1], vbias);
    block[2] = vaddq_f32(block[2], vbias);
    block[3] = vaddq_f32(block[3], vbias);
    for (size_t i = 0; i < row_count; i++) {
        for (size_t j = 0; j < column_count; j++) {
            output[i * output_stride + j] = block[i][j];
        }
    }
    return;
}

void nnp_owt6x6_3x3s2_with_bias__neon(
        const void *restrict transform,
        float output[restrict static 1],
        const float bias[restrict static 1],
        size_t transform_stride,
        size_t output_stride,
        uint32_t row_count,
        uint32_t column_count)
{
    fprintf(stderr, "Error: unsupported kernel nnp_owt6x6_3x3s2_with_bias__neon\n");
    exit(1);
    return;
}

void nnp_owt6x6_3x3_with_bias_with_relu__neon(
        const void *restrict transform,
        float output[restrict static 1],
        const float bias[restrict static 1],
        size_t transform_stride, size_t output_stride,
        uint32_t row_count, uint32_t column_count)
{
    fprintf(stderr, "Error: unsupported kernel nnp_owt6x6_3x3_with_bias_with_relu__neon\n");
    exit(1);
    return;
}

void nnp_owt6x6_3x3s2_with_bias_with_relu__neon(
        const void *restrict transform,
        float output[restrict static 1],
        const float bias[restrict static 1],
        size_t transform_stride, size_t output_stride,
        uint32_t row_count, uint32_t column_count)
{
    fprintf(stderr, "Error: unsupported kernel nnp_owt6x6_3x3s2_with_bias_with_relu__neon\n");
    exit(1);
    return;
}

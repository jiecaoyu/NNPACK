#include <stdint.h>
#include <stdio.h>
#include <stddef.h>
#include <stdlib.h>

#include <nnpack/arm_neon.h>
#include <nnpack/activations.h>
#include <nnpack/macros.h>
#include <nnpack/utils.h>

#include <neon/winograd/f6x6k3x3.h>
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
    fprintf(stderr, "Error: unsupported kernel nnp_iwt6x6_3x3_with_offset__neon\n");
    exit(1);
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
    fprintf(stderr, "Error: unsupported kernel nnp_kwt6x6_3x3__neon\n");
    exit(1);
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
    fprintf(stderr, "Error: unsupported kernel nnp_owt6x6_3x3_with_bias__neon\n");
    exit(1);
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

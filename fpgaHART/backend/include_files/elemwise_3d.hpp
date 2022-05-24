#pragma once

#include "common_.hpp"

/**
 *  ELEMENT WISE ADD FUNCTION
 */
template<
    unsigned int BATCH,
    unsigned int CHANNELS,
    unsigned int HEIGHT,
    unsigned int WIDTH,
    unsigned int DEPTH,
    typename add_t,
	typename accum_add_t
>
void add_3d(
    stream_t(add_t) &in_1,
    stream_t(add_t) &in_2,
    stream_t(add_t) &out
)
{

#pragma HLS INLINE OFF

    // set all parameters as constants
    const unsigned int batch    = BATCH;
    const unsigned int height   = HEIGHT;
    const unsigned int width    = WIDTH;
    const unsigned int depth    = DEPTH;
    const unsigned int channels = CHANNELS;

    for(unsigned long pixel_index=0 ; pixel_index < batch*channels*height*width*depth ; pixel_index++) {
#pragma HLS PIPELINE II=1 rewind
        add_t tmp_1 = in_1.read();
        add_t tmp_2 = in_2.read();
        accum_add_t result = tmp_1 + tmp_2;
        out.write(add_t(result));
    }
}

/**
 *  ELEMENT WISE ADD BROADCAST FUNCTION
 */
template<
    unsigned int BATCH,
    unsigned int CHANNELS,
    unsigned int HEIGHT,
    unsigned int WIDTH,
    unsigned int DEPTH,
    typename add_bc_t,
	typename accum_add_bc_t
>
void add_bc_3d(
    stream_t(add_bc_t) &in_1,
    stream_t(add_bc_t) &in_2,
    stream_t(add_bc_t) &out
)
{

#pragma HLS INLINE OFF

    // set all parameters as constants
    const unsigned int batch    = BATCH;
    const unsigned int height   = HEIGHT;
    const unsigned int width    = WIDTH;
    const unsigned int depth    = DEPTH;
    const unsigned int channels = CHANNELS;

    const unsigned int occurrence_distance = batch*height*width*depth*channels;

    add_bc_t addition[channels];
#pragma HLS ARRAY_PARTITION variable=addition complete dim=0

    for(unsigned long pixel_index=0 ; pixel_index < batch*height*width*depth ; pixel_index++) {
        for(unsigned int channel_index=0 ; channel_index < channels ; channel_index++) {
#pragma HLS LOOP_FLATTEN
#pragma HLS PIPELINE II=1 rewind
            if (pixel_index == 0)
            {
            	DO_PRAGMA(HLS OCCURRENCE cycle=occurrence_distance)
                add_bc_t tmp_2 = in_2.read();
                addition[channel_index] = tmp_2;
            }
            add_bc_t tmp_1 = in_1.read();
            accum_add_bc_t result = addition[channel_index] + tmp_1;
            out.write(add_bc_t(result));
        }
    }
}


/**
 *  ELEMENT WISE MUL FUNCTION
 */
template<
    unsigned int BATCH,
    unsigned int CHANNELS,
    unsigned int HEIGHT,
    unsigned int WIDTH,
    unsigned int DEPTH,
    typename mul_t,
	typename accum_mul_t
>
void mul_3d(
    stream_t(mul_t) &in_1,
    stream_t(mul_t) &in_2,
    stream_t(mul_t) &out
)
{

#pragma HLS INLINE OFF

    // set all parameters as constants
    const unsigned int batch    = BATCH;
    const unsigned int height   = HEIGHT;
    const unsigned int width    = WIDTH;
    const unsigned int depth    = DEPTH;
    const unsigned int channels = CHANNELS;

    for(unsigned long pixel_index=0 ; pixel_index < batch*channels*height*width*depth ; pixel_index++) {
#pragma HLS PIPELINE II=1 rewind
        mul_t tmp_1 = in_1.read();
        mul_t tmp_2 = in_2.read();
        accum_mul_t result = tmp_1 * tmp_2;
        out.write(mul_t(result));
    }
}

/**
 *  ELEMENT WISE MUL BROADCAST FUNCTION
 */
template<
    unsigned int BATCH,
    unsigned int CHANNELS,
    unsigned int HEIGHT,
    unsigned int WIDTH,
    unsigned int DEPTH,
    typename mul_bc_t,
	typename accum_mul_bc_t
>
void mul_bc_3d(
    stream_t(mul_bc_t) &in_1,
    stream_t(mul_bc_t) &in_2,
    stream_t(mul_bc_t) &out
)
{

#pragma HLS INLINE OFF

    // set all parameters as constants
    const unsigned int batch    = BATCH;
    const unsigned int height   = HEIGHT;
    const unsigned int width    = WIDTH;
    const unsigned int depth    = DEPTH;
    const unsigned int channels = CHANNELS;

    const unsigned int occurrence_distance = batch*height*width*depth*channels;

    mul_bc_t multiplication[channels];
#pragma HLS ARRAY_PARTITION variable=multiplication complete dim=0
//#pragma HLS BIND_STORAGE variable=multiplication type=ram_2p

    for(unsigned long pixel_index=0 ; pixel_index < batch*height*width*depth ; pixel_index++) {
        for(unsigned int channel_index=0 ; channel_index < channels ; channel_index++) {
#pragma HLS LOOP_FLATTEN
#pragma HLS PIPELINE II=1 rewind
            if (pixel_index == 0)
            {
            	DO_PRAGMA(HLS OCCURRENCE cycle=occurrence_distance)
                mul_bc_t tmp_2 = in_2.read();
                multiplication[channel_index] = tmp_2;
            }
            mul_bc_t tmp_1 = in_1.read();
            accum_mul_bc_t result = multiplication[channel_index] * tmp_1;
            out.write(mul_bc_t(result));
        }
    }
}







/**
 *  TEST ELEMENT WISE MUL BROADCAST FUNCTION
 */
template<
    unsigned int BATCH,
    unsigned int CHANNELS,
    unsigned int HEIGHT,
    unsigned int WIDTH,
    unsigned int DEPTH,
    typename mul_bc_t,
	typename accum_mul_bc_t
>
void mul_bc_test_3d(
    stream_t(mul_bc_t) &in,
	mul_bc_t multiplication[CHANNELS],
    stream_t(mul_bc_t) &out
)
{

#pragma HLS INLINE OFF

    // set all parameters as constants
    const unsigned int batch    = BATCH;
    const unsigned int height   = HEIGHT;
    const unsigned int width    = WIDTH;
    const unsigned int depth    = DEPTH;
    const unsigned int channels = CHANNELS;

    for(unsigned long pixel_index=0 ; pixel_index < batch*height*width*depth ; pixel_index++) {
        for(unsigned int channel_index=0 ; channel_index < channels ; channel_index++) {
#pragma HLS LOOP_FLATTEN
#pragma HLS PIPELINE II=1 rewind
            mul_bc_t tmp = in.read();
            accum_mul_bc_t result = multiplication[channel_index] * tmp;
            out.write(mul_bc_t(result));
        }
    }
}

/**
 *  ELEMENT WISE MUL BROADCAST FUNCTION
 */
template<
    unsigned int CHANNELS,
    typename mul_bc_t
	>
void mul_bc_copy_3d(
    stream_t(mul_bc_t) &in,
	mul_bc_t multiplication[CHANNELS]
)
{

#pragma HLS INLINE OFF

    // set all parameters as constants
    const unsigned int channels = CHANNELS;

	for(unsigned int channel_index=0 ; channel_index < channels ; channel_index++) {
#pragma HLS PIPELINE II=1 rewind
		mul_bc_t tmp = in.read();
		multiplication[channel_index] = tmp;
	}
}

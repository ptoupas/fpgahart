#pragma once

#include "common_.hpp"

/**
 *  SWISH FUNCTION
 */
template<
    unsigned int BATCH,
    unsigned int CHANNELS,
    unsigned int HEIGHT,
    unsigned int WIDTH,
    unsigned int DEPTH,
    typename swish_t
>
void swish_3d(
    stream_t(swish_t) &in,
    stream_t(swish_t) &out
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
        swish_t tmp = in.read();
    	datamath_t sigmoid_tmp = -datamath_t(tmp);
        swish_t sigmoid_result = swish_t(1) / (swish_t(1) + hls::exp<DATA_WIDTH,INTEGER_BITS>(sigmoid_tmp));
        swish_t result = sigmoid_result * tmp;
        out.write(result);
    }
}

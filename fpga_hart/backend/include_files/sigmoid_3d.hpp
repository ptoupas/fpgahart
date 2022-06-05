#pragma once

#include "common_.hpp"

/**
 *  SIGMOID FUNCTION
 */
template<
    unsigned int BATCH,
    unsigned int CHANNELS,
    unsigned int HEIGHT,
    unsigned int WIDTH,
    unsigned int DEPTH,
    typename sigmoid_t
>
void sigmoid_3d(
    stream_t(sigmoid_t) &in,
    stream_t(sigmoid_t) &out
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
    	datamath_t tmp = -in.read();
        sigmoid_t result = sigmoid_t(1) / (sigmoid_t(1) + hls::exp<DATA_WIDTH,INTEGER_BITS>(tmp));
        out.write(result);
    }
}

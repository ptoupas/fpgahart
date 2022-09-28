#pragma once

#include "common_.hpp"

/**
 *  RECTIFIED LINEAR UNIT (RELU) FUNCTION
 */
template<
    unsigned int BATCH,
    unsigned int CHANNELS,
    unsigned int HEIGHT,
    unsigned int WIDTH,
    unsigned int DEPTH,
    typename relu_t
>
void relu_3d(
    stream_t(relu_t) &in,
    stream_t(relu_t) &out
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
        relu_t tmp = in.read();
        if(tmp < 0.0)
            out.write(0.0);
        else
            out.write(tmp);
    }
}

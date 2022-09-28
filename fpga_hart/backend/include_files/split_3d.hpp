#pragma once

#include "common_.hpp"

/**
 *  SPLIT FUNCTION
 */
template<
    unsigned int BATCH,
    unsigned int CHANNELS,
    unsigned int HEIGHT,
    unsigned int WIDTH,
    unsigned int DEPTH,
    typename split_t
>
void split_3d(
    stream_t(split_t) &in,
    stream_t(split_t) &out1,
    stream_t(split_t) &out2
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
        split_t tmp = in.read();
        out1.write(tmp);
        out2.write(tmp);
    }
}

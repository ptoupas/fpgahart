#pragma once

#include "common_.hpp"

/**
 *  BIAS FUNCTION
 */

template<
    unsigned int BATCH,
    unsigned int DEPTH,
    unsigned int HEIGHT,
    unsigned int WIDTH,
    unsigned int FILTERS,
    typename bias_data_t,
    typename bias_biases_t
>
void bias_3d(
    stream_t(bias_data_t) &in,
    const bias_biases_t bias[FILTERS],
    stream_t(bias_data_t) &out
)
{

#pragma HLS INLINE OFF

    const unsigned int batch        = BATCH;
    const unsigned int height       = HEIGHT;
    const unsigned int width        = WIDTH;
    const unsigned int depth        = DEPTH;
    const unsigned int filters      = FILTERS;

#pragma HLS STREAM variable=in
#pragma HLS STREAM variable=out

    for (unsigned long pixel_index = 0; pixel_index < batch * height * width * depth; pixel_index++)
    {
        filter_loop: for(unsigned int filter_index=0;filter_index<filters;filter_index++)
        {
            #pragma HLS PIPELINE II=1 rewind
            #pragma HLS loop_flatten
            out.write(in.read() + bias[filter_index]);
        }
    }
}
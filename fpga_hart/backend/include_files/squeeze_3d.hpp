#pragma once

#include "common_.hpp"

template<
    unsigned int BATCH,
    unsigned int CHANNELS,
    unsigned int HEIGHT,
    unsigned int WIDTH,
    unsigned int DEPTH,
    unsigned int COARSE_IN,
    unsigned int BUFFER_SIZE,
    typename squeeze_t
>
void squeeze_3d_read_input(
    stream_t(squeeze_t) in[COARSE_IN],
    stream_t(squeeze_t) cache[BUFFER_SIZE]
)
{
#pragma HLS INLINE OFF

    // set all parameters as constants
    const unsigned int batch        = BATCH;
    const unsigned int height       = HEIGHT;
    const unsigned int width        = WIDTH;
    const unsigned int depth        = DEPTH;
    const unsigned int channels     = CHANNELS;
    const unsigned int coarse_in    = COARSE_IN;
    const unsigned int buffer_size  = BUFFER_SIZE;
    const unsigned int channels_per_buffer  = DIVIDE(channels,buffer_size);

    dim_in_loop: for (unsigned int pixel_index = 0; pixel_index < batch*height*width*depth*channels_per_buffer; pixel_index++) {
        unsigned int cache_in_index = 0;
        channel_in_loop: for (unsigned int channel_index = 0; channel_index < DIVIDE(buffer_size,coarse_in); channel_index++) {
            #pragma HLS loop_flatten
            #pragma HLS pipeline II=1 rewind
            in_loop: for (unsigned int in_index = 0; in_index < coarse_in; in_index++) {
                cache[cache_in_index+in_index].write(in[in_index].read());
            }
            cache_in_index += coarse_in;
        }
    }
}

template<
    unsigned int BATCH,
    unsigned int CHANNELS,
    unsigned int HEIGHT,
    unsigned int WIDTH,
    unsigned int DEPTH,
    unsigned int COARSE_OUT,
    unsigned int BUFFER_SIZE,
    typename squeeze_t
>
void squeeze_3d_write_output(
    stream_t(squeeze_t) cache[BUFFER_SIZE],
    stream_t(squeeze_t) out[COARSE_OUT]
)
{

#pragma HLS INLINE OFF

    // set all parameters as constants
    const unsigned int batch        = BATCH;
    const unsigned int height       = HEIGHT;
    const unsigned int width        = WIDTH;
    const unsigned int depth        = DEPTH;
    const unsigned int channels     = CHANNELS;
    const unsigned int coarse_out   = COARSE_OUT;
    const unsigned int buffer_size  = BUFFER_SIZE;
    const unsigned int channels_per_buffer  = DIVIDE(channels,buffer_size);

    dim_out_loop: for (unsigned int pixel_index = 0; pixel_index < batch*height*width*depth*channels_per_buffer; pixel_index++) {
        unsigned int cache_out_index = 0;
        channel_out_loop: for (unsigned int channel_index = 0; channel_index < DIVIDE(buffer_size,coarse_out); channel_index++) {
            #pragma HLS loop_flatten
            #pragma HLS pipeline II=1 rewind
            out_loop: for (unsigned int out_index = 0; out_index < coarse_out; out_index++) {
                out[out_index].write(cache[cache_out_index+out_index].read());
            }
            cache_out_index += coarse_out;
        }
    }
}

template<
    unsigned int BATCH,
    unsigned int CHANNELS,
    unsigned int HEIGHT,
    unsigned int WIDTH,
    unsigned int DEPTH,
    unsigned int COARSE_IN,
    unsigned int COARSE_OUT,
    unsigned int BUFFER_SIZE,
    typename squeeze_t
>
void squeeze_3d(
    stream_t(squeeze_t) in[COARSE_IN],
    stream_t(squeeze_t) out[COARSE_OUT]
)
{

#pragma HLS INLINE OFF
#pragma HLS DATAFLOW

//    // set all parameters as constants
    const unsigned int buffer_size  = BUFFER_SIZE;

#pragma HLS ARRAY_PARTITION variable=in complete dim=0
#pragma HLS ARRAY_PARTITION variable=out complete dim=0

    stream_t(squeeze_t)  cache[buffer_size];
#pragma HLS STREAM variable=cache
#pragma HLS ARRAY_PARTITION variable=cache complete dim=0

    squeeze_3d_read_input<
    			BATCH,
				CHANNELS,
				HEIGHT,
				WIDTH,
				DEPTH,
				COARSE_IN,
				BUFFER_SIZE,
				squeeze_t
	           >(in, cache);


    squeeze_3d_write_output<
        			BATCH,
    				CHANNELS,
    				HEIGHT,
    				WIDTH,
    				DEPTH,
    				COARSE_OUT,
    				BUFFER_SIZE,
    				squeeze_t
    		      >(cache, out);

}

#pragma once

#include "common_.hpp"

/**
 *  GLOBAL AVERAGE POOLING FUNCTION
 */
template<
    unsigned int BATCH,
    unsigned int CHANNELS,
    unsigned int HEIGHT,
    unsigned int WIDTH,
    unsigned int DEPTH,
    typename gap_t,
    typename avg_gap_t
>
void gap_3d(
    stream_t(gap_t) &in,
    stream_t(gap_t) &out
)
{
#pragma HLS INLINE OFF

    // set all parameters as constants
    const unsigned int batch    	= BATCH;
    const unsigned int height   	= HEIGHT;
    const unsigned int width    	= WIDTH;
    const unsigned int depth    	= DEPTH;
    const unsigned int channels 	= CHANNELS;
	
	const unsigned int feature_map_dimensions = height*width*depth;

	avg_gap_t reader[channels];
#pragma HLS ARRAY_PARTITION variable=reader complete dim=0
//#pragma HLS BIND_STORAGE variable=reader type=ram_2p

	read_loop_pixel: for(unsigned long pixel_index=0 ; pixel_index < batch*height*width*depth ; pixel_index++) {
		read_loop_channel: for(unsigned int channel_index=0 ; channel_index < channels ; channel_index++) {
#pragma HLS PIPELINE II=1 rewind
			gap_t tmp = in.read();
			if (pixel_index == 0){
				reader[channel_index] = tmp;
			}else{
				reader[channel_index] += tmp;
			}
		}
	}

	average_loop_channel: for(unsigned int channel_index=0 ; channel_index < channels ; channel_index++) {
#pragma HLS PIPELINE II=1 rewind
		avg_gap_t tmp = reader[channel_index];
		gap_t result = gap_t(tmp) / gap_t(feature_map_dimensions);
		out.write(gap_t(result));
	}

}

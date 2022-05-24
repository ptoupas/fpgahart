#pragma once

#include "common_.hpp"

/**
 *  GLUE FUNCTION
 */
template<
	unsigned int BATCH,
	unsigned int FILTERS,
	unsigned int HEIGHT,
	unsigned int WIDTH,
	unsigned int DEPTH,
	unsigned int COARSE_IN,
	unsigned int COARSE_OUT,
    typename glue_acc_t,
    typename glue_data_t
>
void glue_3d(
		hls::stream<glue_acc_t> in[COARSE_IN][COARSE_OUT],
		hls::stream<glue_data_t> out[COARSE_OUT]
)
{

#pragma HLS INLINE OFF

    // set all parameters as constants
    const unsigned int batch    	= BATCH;
    const unsigned int height   	= HEIGHT;
    const unsigned int width    	= WIDTH;
    const unsigned int depth    	= DEPTH;
    const unsigned int filters  	= FILTERS;
    const unsigned int coarse_in 	= COARSE_IN;
    const unsigned int coarse_out	= COARSE_OUT;

    const unsigned int filters_per_coarse = DIVIDE(filters,coarse_out);

    glue_acc_t acc[coarse_out];
#pragma HLS ARRAY_PARTITION variable=acc complete dim=0

    pixel_loop: for(unsigned long pixel_index=0;pixel_index<batch*height*width*depth;pixel_index++) {
			filter_loop: for(unsigned int filter_index=0;filter_index<filters_per_coarse;filter_index++) {
#pragma HLS LOOP_FLATTEN
#pragma HLS PIPELINE II=1
//				#pragma HLS unroll region
//				coarse_group_loop: for(unsigned int group_index=0; group_index<coarse_group; group_index++) {
					coarse_out_loop: for(unsigned int out_index=0; out_index<coarse_out; out_index++) {
						coarse_in_loop: for(unsigned int in_index=0; in_index<coarse_in; in_index++) {
							// update accumulation cache
							glue_acc_t prev = ( in_index == 0 ) ? glue_acc_t(0) : acc[out_index] ;
#ifdef DEBUG_PRINTS
							glue_acc_t tmp = in[in_index][out_index].read();
							acc[out_index] = prev + tmp ;
							cout << "read in[" << in_index << "][" << out_index << "] = " << tmp << ". Previous val = " << prev << endl;
							cout << "acc[" << out_index << "] = " << acc[out_index] << endl;
#else
							acc[out_index] = prev + in[in_index][out_index].read() ;
#endif
							// write to output stream
							if( in_index == (coarse_in-1) ) {
#ifdef DEBUG_PRINTS
								cout << "WRITE out[" << out_index << "] = " << acc[out_index] << endl;
#endif
								out[out_index].write( glue_data_t(acc[out_index]) ) ;
							}
						}
					}
//				}
			}
    }
}



/**
 *  GLUE DEPTHWISE FUNCTION
 */
template<
	unsigned int BATCH,
	unsigned int CHANNELS,
	unsigned int FILTERS,
	unsigned int HEIGHT,
	unsigned int WIDTH,
	unsigned int DEPTH,
	unsigned int GROUPS,
	unsigned int COARSE_IN,
	unsigned int COARSE_OUT,
    typename glue_acc_t,
    typename glue_data_t
>
void glue_dw_3d(
		hls::stream<glue_acc_t> in[COARSE_IN][COARSE_OUT],
		hls::stream<glue_data_t> out[COARSE_IN]
)
{

#pragma HLS INLINE OFF

    // set all parameters as constants
    const unsigned int batch    	= BATCH;
    const unsigned int height   	= HEIGHT;
    const unsigned int width    	= WIDTH;
    const unsigned int depth    	= DEPTH;
    const unsigned int filters  	= FILTERS;
	const unsigned int channels 	= CHANNELS;
	const unsigned int groups   	= GROUPS;
	const unsigned int coarse_in 	= COARSE_IN;
    const unsigned int coarse_out	= COARSE_OUT;

	const unsigned int filters_per_group  = DIVIDE(filters ,groups);

    pixel_loop: for(unsigned long pixel_index=0;pixel_index<batch*height*width*depth;pixel_index++) {
    	channel_loop: for(unsigned int channel_index=0;channel_index<channels;channel_index++) {
			filter_loop: for(unsigned int filter_index=0;filter_index<filters_per_group;filter_index++) {
#pragma HLS LOOP_FLATTEN
#pragma HLS PIPELINE II=1 rewind
				coarse_out_loop: for(unsigned int out_index=0; out_index<coarse_out; out_index++) {
					coarse_in_loop: for(unsigned int in_index=0; in_index<coarse_in; in_index++) {
						// update accumulation cache
						glue_acc_t tmp = in[in_index][out_index].read();
#ifdef DEBUG_PRINTS
						cout << "read in[" << in_index << "][" << out_index << "] = " << tmp << endl;
						cout << "WRITE out[" << out_index << "] = " << tmp << endl;
#endif
						// write to output stream
						out[in_index].write( glue_data_t(tmp) ) ;
					}
				}
			}
    	}
    }
}

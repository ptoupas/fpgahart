#pragma once

#include "common_.hpp"

template<
	unsigned int BATCH,
	unsigned int CHANNELS,
	unsigned int FILTERS,
	unsigned int HEIGHT,
	unsigned int WIDTH,
	unsigned int DEPTH,
	unsigned int K_H,
	unsigned int K_W,
	unsigned int K_D,
	unsigned int FINE,
	unsigned int GROUPS,
    typename conv_data_t,
    typename conv_weight_t
>
void conv_3d_intr(
	hls::stream<conv_data_t>    in[K_H][K_W][K_D],
    const conv_weight_t      	weights[CHANNELS][DIVIDE(FILTERS,GROUPS)][K_H][K_W][K_D],
	hls::stream<conv_data_t>    window_stream[FINE],
	hls::stream<conv_weight_t>  weight_stream[FINE]
)
{

#pragma HLS INLINE OFF
#pragma HLS STABLE variable=weights

    // set all parameters as constants
    const unsigned int batch    = BATCH;
    const unsigned int height   = HEIGHT;
    const unsigned int width    = WIDTH;
    const unsigned int depth    = DEPTH;
    const unsigned int channels = CHANNELS;
    const unsigned int filters  = FILTERS;
    const unsigned int groups   = GROUPS;
	const unsigned int kh   	= K_H;
	const unsigned int kw   	= K_W;
	const unsigned int kd   	= K_D;
	const unsigned int fine   	= FINE;

	const unsigned int channels_per_group = channels; //DIVIDE(channels, groups);
	const unsigned int filters_per_group  = DIVIDE(filters, groups);

    const unsigned int interval = DIVIDE(kh*kw*kd,fine);
	
	const unsigned int window_cache_dependence_distance = batch*height*width*depth*channels;

    const unsigned int weights_partition_factor_k1 = MIN(fine,kh);
    const unsigned int weights_partition_factor_k2 = (fine<=kh) ? 1 : kw;
    const unsigned int weights_partition_factor_k3 = (fine<=kh*kw) ? 1 : kd;

    DO_PRAGMA(HLS ARRAY_PARTITION variable=weights block factor=weights_partition_factor_k1 dim=3)
	DO_PRAGMA(HLS ARRAY_PARTITION variable=weights block factor=weights_partition_factor_k2 dim=4)
	DO_PRAGMA(HLS ARRAY_PARTITION variable=weights block factor=weights_partition_factor_k3 dim=5)

    // INTERLEAVING LOOP
    conv_data_t window_cache[kh][kw][kd];
    #pragma HLS ARRAY_PARTITION variable=window_cache complete dim=0


    intr_pixel_loop: for(unsigned int pixel_index=0;pixel_index<batch*height*width*depth;pixel_index++) {
		intr_channel_loop: for(unsigned int channel_index=0;channel_index<channels_per_group;channel_index++) {
			intr_filter_loop: for(unsigned int filter_index=0;filter_index<filters_per_group;filter_index++) {

#pragma HLS LOOP_FLATTEN
		DO_PRAGMA(HLS PIPELINE II=interval)
#pragma HLS DEPENDENCE variable=window_cache RAW intra true
		DO_PRAGMA(HLS DEPENDENCE variable=window_cache WAW inter true distance=window_cache_dependence_distance )

				unsigned char fine_index = 0;
#ifdef DEBUG_PRINTS
				printf("================================== ");
				printf("Pixel: %d, channel: %d, filter: %d\n", pixel_index, channel_index, filter_index);
#endif
				intr_k3_loop: for(unsigned char k3=0;k3<kd;k3++) {
					intr_k2_loop: for(unsigned char k2=0;k2<kw;k2++) {
						intr_k1_loop: for(unsigned char k1=0;k1<kh;k1++) {
							if(filter_index%filters_per_group == 0) {
								DO_PRAGMA(HLS OCCURRENCE cycle=window_cache_dependence_distance)
								window_cache[k1][k2][k3] = in[k1][k2][k3].read();
#ifdef DEBUG_PRINTS
								printf("INPUT[%d][%d][%d]=%f, ", k1,k2,k3,window_cache[k1][k2][k3].to_float());
#endif
							}
#ifdef DEBUG_PRINTS
							printf("CACHE[%d][%d][%d]=%f, ", k1,k2,k3,window_cache[k1][k2][k3].to_float());
							printf("WEIGHTS[%d][%d][%d][%d][%d]=%f, fine_idx=%d\n", channel_index,filter_index,k1,k2,k3,weights[channel_index][filter_index][k1][k2][k3].to_float(), fine_index);
#endif
							window_stream[fine_index].write(window_cache[k1][k2][k3]);
							weight_stream[fine_index].write(weights[channel_index][filter_index][k1][k2][k3]);

							fine_index = ( fine_index + 1 ) % fine;
						}
					}
				}
			}
		}
    }
}

template<
	unsigned int BATCH,
	unsigned int CHANNELS,
	unsigned int FILTERS,
	unsigned int HEIGHT,
	unsigned int WIDTH,
	unsigned int DEPTH,
	unsigned int K_H,
	unsigned int K_W,
	unsigned int K_D,
	unsigned int FINE,
	unsigned int GROUPS,
	typename conv_data_t,
	typename conv_weight_t,
	typename conv_acc_t
>
void conv_3d_mul(
		hls::stream<conv_data_t> window_stream[FINE],
		hls::stream<conv_weight_t> weight_stream[FINE],
		hls::stream<conv_acc_t> acc_stream[FINE]
)
{

#pragma HLS INLINE OFF

    // set all parameters as constants
    const unsigned int batch    = BATCH;
    const unsigned int height   = HEIGHT;
    const unsigned int width    = WIDTH;
    const unsigned int depth    = DEPTH;
    const unsigned int channels = CHANNELS;
    const unsigned int filters  = FILTERS;
    const unsigned int groups   = GROUPS;
	const unsigned int kh   	= K_H;
	const unsigned int kw   	= K_W;
	const unsigned int kd   	= K_D;
	const unsigned int fine   	= FINE;

	const unsigned int channels_per_group = channels; //DIVIDE(channels, groups);
	const unsigned int filters_per_group  = DIVIDE(filters, groups);

    const unsigned int interval = DIVIDE(kh*kw*kd,fine);

    // MULTIPLICATION LOOP
    conv_acc_t acc_cache[fine];
    unsigned int acc_index=0;
    mul_pixel_loop: for(unsigned int pixel_index=0;pixel_index<batch*height*width*depth*channels_per_group*filters_per_group*interval;pixel_index++) {
#pragma HLS LOOP_FLATTEN
#pragma HLS PIPELINE II=1
            mul_loop: for(unsigned char fine_index=0;fine_index<fine;fine_index++) {
                // update accumulation cache
                conv_acc_t prev = ( acc_index == 0 ) ? conv_acc_t(0) : acc_cache[fine_index] ;
                conv_weight_t weight = weight_stream[fine_index].read();
                conv_data_t data = window_stream[fine_index].read();
#ifdef DEBUG_PRINTS
                printf("acc_idx: %d, fine_idx: %d, res = %f + %f*%f\n", acc_index, fine_index, prev.to_float(), data.to_float(), weight.to_float());
#endif
                acc_cache[fine_index] = prev + data * weight;
                // write to output stream
                if( acc_index == (interval-1) ) {
#ifdef DEBUG_PRINTS
                		printf("acc_stream [%d] <- %f\n", fine_index, acc_cache[fine_index].to_float());
#endif
                    acc_stream[fine_index].write( acc_cache[fine_index] ) ;
                }
            }
            // increment accumulation index
            acc_index = (acc_index+1) % interval;
    }
}

template<
	unsigned int BATCH,
	unsigned int CHANNELS,
	unsigned int FILTERS,
	unsigned int HEIGHT,
	unsigned int WIDTH,
	unsigned int DEPTH,
	unsigned int FINE,
	unsigned int GROUPS,
    typename conv_acc_t
>
void conv_3d_acc(
		hls::stream<conv_acc_t> acc_stream[FINE],
		hls::stream<conv_acc_t> &out
)
{

#pragma HLS INLINE OFF

    // set all parameters as constants
    const unsigned int batch    = BATCH;
    const unsigned int height   = HEIGHT;
    const unsigned int width    = WIDTH;
    const unsigned int depth    = DEPTH;
    const unsigned int channels = CHANNELS;
    const unsigned int filters  = FILTERS;
    const unsigned int groups   = GROUPS;
	const unsigned int fine   	= FINE;

	const unsigned int channels_per_group = channels; //DIVIDE(channels,groups);
	const unsigned int filters_per_group  = DIVIDE(filters ,groups);

    // ACCUMULATION LOOP
    acc_pixel_loop: for(unsigned int pixel_index=0;pixel_index<batch*height*width*depth*channels_per_group*filters_per_group;pixel_index++) {
#pragma HLS PIPELINE II=1 rewind
//        #pragma HLS unroll region
        conv_acc_t acc = 0 ;
        acc_fine_loop: for(unsigned char fine_index=0;fine_index<fine;fine_index++) {
        		conv_acc_t tmp = acc_stream[fine_index].read();
#ifdef DEBUG_PRINTS
        		printf("fine_idx: %d, acc = %f + %f\n", fine_index, acc.to_float(), tmp.to_float());
#endif
        		acc += tmp;
        }
#ifdef DEBUG_PRINTS
        printf("WRITE OUT acc = %f\n",acc.to_float());
#endif
        out.write(acc);
    }
}



template<
	unsigned int BATCH,
	unsigned int CHANNELS,
	unsigned int FILTERS,
	unsigned int HEIGHT,
	unsigned int WIDTH,
	unsigned int DEPTH,
	unsigned int K_H,
	unsigned int K_W,
	unsigned int K_D,
	unsigned int FINE,
	unsigned int GROUPS,
    typename conv_data_t,
    typename conv_weight_t,
    typename conv_acc_t
>
void conv_3d(
		hls::stream<conv_data_t> in[K_H][K_W][K_D],
		const conv_weight_t weights[CHANNELS][DIVIDE(FILTERS,GROUPS)][K_H][K_W][K_D],
		hls::stream<conv_acc_t> &out
)
{

#pragma HLS INLINE OFF

#pragma HLS DATAFLOW

#pragma HLS STABLE variable=weights

    const unsigned char fine = FINE;

    hls::stream<conv_data_t> window_stream[fine];
    hls::stream<conv_weight_t> weight_stream[fine];
    hls::stream<conv_acc_t> acc_stream[fine];

#pragma HLS STREAM variable=window_stream
#pragma HLS ARRAY_PARTITION variable=window_stream complete dim=0
//#pragma HLS BIND_STORAGE variable=window_stream type=fifo impl=bram

#pragma HLS STREAM variable=weight_stream
#pragma HLS ARRAY_PARTITION variable=weight_stream complete dim=0
//#pragma HLS BIND_STORAGE variable=weight_stream type=fifo impl=bram

#pragma HLS STREAM variable=acc_stream
#pragma HLS ARRAY_PARTITION variable=acc_stream complete dim=0
//#pragma HLS BIND_STORAGE variable=acc_stream type=fifo impl=bram


    conv_3d_intr<
				BATCH,
				CHANNELS,
				FILTERS,
				HEIGHT,
				WIDTH,
				DEPTH,
				K_H,
				K_W,
				K_D,
				FINE,
				GROUPS,
				conv_data_t,
				conv_weight_t>
    (in,weights,window_stream,weight_stream);


    conv_3d_mul<
				BATCH,
				CHANNELS,
				FILTERS,
				HEIGHT,
				WIDTH,
				DEPTH,
				K_H,
				K_W,
				K_D,
				FINE,
				GROUPS,
				conv_data_t,
				conv_weight_t,
				conv_acc_t>
    (window_stream,weight_stream,acc_stream);


    conv_3d_acc<
				BATCH,
				CHANNELS,
				FILTERS,
				HEIGHT,
				WIDTH,
				DEPTH,
				FINE,
				GROUPS,
				conv_acc_t>
    (acc_stream, out);

}


/**
 *  POINTWISE CONVOLUTION FUNCTION
 */
template<
	unsigned int BATCH,
	unsigned int CHANNELS,
	unsigned int FILTERS,
	unsigned int HEIGHT,
	unsigned int WIDTH,
	unsigned int DEPTH,
	unsigned int GROUPS,
    typename conv_data_t,
    typename conv_weight_t,
    typename conv_acc_t
>
void conv_3d(
		hls::stream<conv_data_t> &in,
		const conv_weight_t weights[CHANNELS][DIVIDE(FILTERS,GROUPS)][1][1][1],
		hls::stream<conv_acc_t> &out
)
{

#pragma HLS INLINE OFF
#pragma HLS STABLE variable=weights

    // set all parameters as constants
    const unsigned int batch    = BATCH;
    const unsigned int height   = HEIGHT;
    const unsigned int width    = WIDTH;
    const unsigned int depth    = DEPTH;
    const unsigned int channels = CHANNELS;
    const unsigned int filters  = FILTERS;
    const unsigned int groups   = GROUPS;
	
	const unsigned int occurrence_filter_distance = batch*height*width*depth*channels;

	const unsigned int channels_per_group = channels; //DIVIDE(channels,groups);
	const unsigned int filters_per_group  = DIVIDE(filters ,groups);

    conv_data_t window_cache;

    pixel_loop: for(unsigned int pixel_index=0;pixel_index<batch*height*width*depth;pixel_index++) {
        channel_loop: for(unsigned int channel_index=0;channel_index<channels_per_group;channel_index++) {
            filter_loop: for(unsigned int filter_index=0;filter_index<filters_per_group;filter_index++) {
#pragma HLS LOOP_FLATTEN
#pragma HLS PIPELINE II=1
//#pragma HLS DEPENDENCE variable=windowCache RAW intra true
                if(filter_index%filters_per_group == 0) {
                	DO_PRAGMA(HLS OCCURRENCE cycle=occurrence_filter_distance)
                    window_cache = in.read();
                }

                conv_acc_t acc = window_cache * weights[channel_index][filter_index][0][0][0];

                out.write(acc);
            }
        }
    }
}

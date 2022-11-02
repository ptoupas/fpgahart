#pragma once

#include "common_.hpp"

template<
    unsigned int BATCH,
    unsigned int CHANNELS,
    unsigned int FILTERS,
    unsigned int HEIGHT,
    unsigned int WIDTH,
	unsigned int DEPTH,
	unsigned int GROUPS,
    typename T
>
void accum_3d(
		hls::stream<T> &in,
		hls::stream<T> &out
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

	const unsigned int channels_per_group = DIVIDE(channels,groups);
	const unsigned int filters_per_group  = DIVIDE(filters ,groups);

#pragma HLS STREAM variable=in
#pragma HLS STREAM variable=out depth=filters_per_group*2+1

    T acc[filters_per_group];
//#pragma HLS BIND_STORAGE variable=acc type=ram_2p

    stream_pixel_loop: for(unsigned long pixel_index=0;pixel_index<batch*height*width*depth*groups;pixel_index++) {
        stream_channel_loop: for(unsigned int channel_index=0;channel_index<channels_per_group;channel_index++) {
            stream_filter_loop: for(unsigned int filter_index=0;filter_index<filters_per_group;filter_index++) {
                #pragma HLS PIPELINE II=1 rewind
                #pragma HLS DEPENDENCE variable=acc WAR intra true
                T prev = ( channel_index == 0 ) ? T(0) : acc[filter_index] ;
                T curr =  prev + in.read();
                acc[filter_index] = curr;

                if( channel_index == (channels_per_group-1) ) {
                    out.write( curr ) ;
                }
            }
        }
    }
}



template<
    unsigned int CHANNELS,
    unsigned int FILTERS,
    unsigned int GROUPS,
    typename accum_t
>
void accum_accumulate_inner_3d(
    stream_t(accum_t) in[DIVIDE(FILTERS, GROUPS)],
    stream_t(accum_t) &out,
    accum_t acc,
    unsigned int filter_index
) {

#pragma HLS INLINE OFF

    const unsigned int channels   = CHANNELS;
    const unsigned int filters    = FILTERS;
    const unsigned int groups     = GROUPS;
    const unsigned int channels_per_group = DIVIDE(channels,groups);
    const unsigned int filters_per_group  = DIVIDE(filters ,groups);

    channel_loop: for(unsigned int channel_index=0; channel_index<channels_per_group; channel_index++) {
        #pragma HLS loop_flatten
        #pragma HLS pipeline II=1 rewind
        #pragma HLS dependence variable=acc WAR intra false
        accum_t cache = in[filter_index].read();
        acc = ( channel_index == 0 ) ?  cache : accum_t(cache + acc);
        if( channel_index == (channels_per_group-1) ) {
            out.write( acc ) ;
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
	unsigned int GROUPS,
	typename accum_t
>
void accum_accumulate_3d(
    stream_t(accum_t) in[DIVIDE(FILTERS, GROUPS)],
    stream_t(accum_t) &out
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

	const unsigned int filters_per_group  = DIVIDE(filters ,groups);

    // accumulation cache
    accum_t acc = 0;
    // #pragma HLS dependence variable=acc WAR intra false
    // #pragma HLS dependence variable=acc RAW intra true

    pixel_loop: for(unsigned int pixel_index=0; pixel_index<batch*height*width*depth*groups; pixel_index++) {
        filter_loop: for(unsigned int filter_index=0; filter_index<filters_per_group; filter_index++) {
            accum_accumulate_inner_3d<CHANNELS, FILTERS, GROUPS, accum_t>(in, out, acc, filter_index);
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
	unsigned int GROUPS,
	typename accum_t
>
void accum_reorder_3d(
    stream_t(accum_t) &in,
    stream_t(accum_t) out[DIVIDE(FILTERS, GROUPS)]
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

	const unsigned int filters_per_group  = DIVIDE(filters ,groups);

    pixel_loop: for(unsigned int pixel_index=0; pixel_index<batch*height*width*depth*channels; pixel_index++) {
        filter_loop: for(unsigned int filter_index=0; filter_index<filters_per_group; filter_index++) {
            #pragma HLS loop_flatten
            #pragma HLS pipeline II=1 rewind
            out[filter_index].write( in.read() );
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
	unsigned int GROUPS,
    typename accum_t
>
void accum_3d_new(
    stream_t(accum_t) &in,
    stream_t(accum_t) &out
)
{

    #pragma HLS INLINE OFF
    #pragma HLS DATAFLOW

    // get all constant parameters
	const unsigned int channels   = CHANNELS;
    const unsigned int filters    = FILTERS;
    const unsigned int groups     = GROUPS;
    const unsigned int channels_per_group = DIVIDE(channels,groups);
    const unsigned int filters_per_group  = DIVIDE(filters ,groups);

    #pragma HLS STREAM variable=in
    #pragma HLS STREAM variable=out

    // re-ordered stream
    stream_t(accum_t) reorder[filters_per_group];
    #pragma HLS array_partition variable=reorder complete dim=0
    DO_PRAGMA(HLS STREAM variable=reorder depth=channels_per_group+1)

    accum_reorder_3d<
		BATCH,
        CHANNELS,
        FILTERS,
		HEIGHT,
		WIDTH,
		DEPTH,
        GROUPS,
        accum_t
    >(in, reorder);

    accum_accumulate_3d<
		BATCH,
		CHANNELS,
		FILTERS,
		HEIGHT,
		WIDTH,
		DEPTH,
		GROUPS,
		accum_t
    >(reorder, out);

}

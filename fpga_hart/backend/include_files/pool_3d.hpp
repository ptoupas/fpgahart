#pragma once

#include "common_.hpp"

/**
  * POOL 3D
  */

template<
    unsigned int K_H,
	unsigned int K_W,
	unsigned int K_D,
    typename pool_t
>
void pool_3d_inner(
    stream_t(pool_t) in[K_H][K_W][K_D],
    stream_t(pool_t) &out
)
{

#pragma HLS INLINE OFF

	const unsigned int kh   	= K_H;
	const unsigned int kw   	= K_W;
	const unsigned int kd   	= K_D;

#pragma HLS STREAM variable=in
#pragma HLS STREAM variable=out
#pragma HLS ARRAY_PARTITION variable=in complete dim=0

    pool_t cache;
    #pragma HLS DEPENDENCE variable=cache RAW intra true

    #pragma HLS unroll region
    pool_loop_1: for (unsigned char k1 = 0; k1 < kh; k1++) {
        pool_loop_2: for (unsigned char k2 = 0; k2 < kw; k2++) {
            pool_loop_3: for (unsigned char k3 = 0; k3 < kd; k3++) {
                if (k1 == 0 && k2 == 0 && k3 == 0) {
                    cache = in[k1][k2][k3].read();
                }
                else {
                    pool_t tmp = in[k1][k2][k3].read();
                    cache = (cache > tmp ) ? cache : tmp ;
                }
            }
        }
    }
    out.write(cache);
}

template<
    unsigned int BATCH,
    unsigned int CHANNELS,
    unsigned int HEIGHT,
    unsigned int WIDTH,
    unsigned int DEPTH,
    unsigned int K_H,
    unsigned int K_W,
    unsigned int K_D,
    typename pool_t
>
void pool_3d(
    stream_t(pool_t) in[K_H][K_W][K_D],
    stream_t(pool_t) &out
)
{

#pragma HLS INLINE OFF

    const unsigned int batch_size   = BATCH;
    const unsigned int height       = HEIGHT;
    const unsigned int width        = WIDTH;
    const unsigned int depth        = DEPTH;
    const unsigned int channels     = CHANNELS;

#pragma HLS STREAM variable=in
#pragma HLS STREAM variable=out
#pragma HLS ARRAY_PARTITION variable=in complete dim=0

    pixel_loop: for(unsigned long pixel_index=0;pixel_index<batch_size*height*width*depth*channels;pixel_index++) {
        #pragma HLS PIPELINE II=1 rewind
        pool_3d_inner<K_H,K_W,K_D,pool_t>(in,out);
    }
}

template<
    unsigned int K_H,
    unsigned int K_W,
    unsigned int K_D,
    typename pool_t
>
void pool_3d(
    stream_t(pool_t) in[K_H][K_W][K_D],
    stream_t(pool_t) &out
)
{

#pragma HLS INLINE OFF

#pragma HLS STREAM variable=in
#pragma HLS STREAM variable=out
#pragma HLS ARRAY_PARTITION variable=in complete dim=0

    #pragma HLS PIPELINE II=1 rewind
    pool_3d_inner<K_H,K_W,K_D,pool_t>(in,out);
}

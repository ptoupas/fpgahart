#pragma once

#include "common_.hpp"

template <
    unsigned int BATCH,
    unsigned int CHANNELS,
    unsigned int HEIGHT,
    unsigned int WIDTH,
    unsigned int DEPTH,
    unsigned int COARSE_OUT,
    unsigned int K_H,
    unsigned int K_W,
    unsigned int K_D,
    typename T>
void fork_3d(hls::stream<T> in[K_H][K_W][K_D], hls::stream<T> out[COARSE_OUT][K_H][K_W][K_D])
{

#pragma HLS INLINE OFF

    // set all parameters as constants
    const unsigned int batch = BATCH;
    const unsigned int height = HEIGHT;
    const unsigned int width = WIDTH;
    const unsigned int depth = DEPTH;
    const unsigned int channels = CHANNELS;
    const unsigned int kh = K_H;
    const unsigned int kw = K_W;
    const unsigned int kd = K_D;
    const unsigned int coarse_out = COARSE_OUT;

    const unsigned int occurrence_coarse_distance = batch * height * width * depth * channels * kh * kw * kd;

    T local_cache[kh][kw][kd];
#pragma HLS ARRAY_PARTITION variable = local_cache complete dim = 0
#pragma HLS DEPENDENCE variable = local_cache RAW intra true

pixel_loop:
    for (unsigned long pixel_index = 0; pixel_index < batch * height * width * depth * channels; pixel_index++)
    {
#pragma HLS PIPELINE II = 1 rewind
    k1_loop:
        for (unsigned char k1 = 0; k1 < kh; k1++)
        {
        k2_loop:
            for (unsigned char k2 = 0; k2 < kw; k2++)
            {
            k3_loop:
                for (unsigned char k3 = 0; k3 < kd; k3++)
                {
                coarse_loop:
                    for (unsigned int coarse_index = 0; coarse_index < coarse_out; coarse_index++)
                    {
                        if (coarse_index == 0)
                        {
                            DO_PRAGMA(HLS OCCURRENCE cycle = occurrence_coarse_distance)
                            local_cache[k1][k2][k3] = in[k1][k2][k3].read();
                        }
                        out[coarse_index][k1][k2][k3].write(local_cache[k1][k2][k3]);
                    }
                }
            }
        }
    }
}

template <
    unsigned int BATCH,
    unsigned int IN_FEATURES,
    unsigned int COARSE_OUT,
    typename T>
void fork_3d(hls::stream<T> &in, hls::stream<T> out[COARSE_OUT])
{

#pragma HLS INLINE OFF

    // set all parameters as constants
    const unsigned int batch = BATCH;
    const unsigned int in_features = IN_FEATURES;
    const unsigned int coarse_out = COARSE_OUT;

    const unsigned int occurrence_coarse_distance = batch * in_features;

    T local_cache;

pixel_loop:
    for (unsigned long pixel_index = 0; pixel_index < batch * in_features; pixel_index++)
    {
#pragma HLS PIPELINE II = 1 rewind
    coarse_loop:
        for (unsigned int coarse_index = 0; coarse_index < coarse_out; coarse_index++)
        {
            if (coarse_index == 0)
            {
                DO_PRAGMA(HLS OCCURRENCE cycle = occurrence_coarse_distance)
                local_cache = in.read();
            }
            out[coarse_index].write(local_cache);
        }
    }
}

template <
    unsigned int BATCH,
    unsigned int CHANNELS,
    unsigned int HEIGHT,
    unsigned int WIDTH,
    unsigned int DEPTH,
    unsigned int COARSE_OUT,
    typename T>
void fork_3d(hls::stream<T> &in, hls::stream<T> out[COARSE_OUT])
{

#pragma HLS INLINE OFF

    // set all parameters as constants
    const unsigned int batch = BATCH;
    const unsigned int height = HEIGHT;
    const unsigned int width = WIDTH;
    const unsigned int depth = DEPTH;
    const unsigned int channels = CHANNELS;
    const unsigned int coarse_out = COARSE_OUT;

    const unsigned int occurrence_coarse_distance = batch * height * width * depth * channels;

    T local_cache;

pixel_loop:
    for (unsigned long pixel_index = 0; pixel_index < batch * height * width * depth * channels; pixel_index++)
    {
#pragma HLS PIPELINE II = 1 rewind
    coarse_loop:
        for (unsigned int coarse_index = 0; coarse_index < coarse_out; coarse_index++)
        {
            if (coarse_index == 0)
            {
                DO_PRAGMA(HLS OCCURRENCE cycle = occurrence_coarse_distance)
                local_cache = in.read();
            }
            out[coarse_index].write(local_cache);
        }
    }
}

template <
    unsigned int BATCH,
    unsigned int CHANNELS,
    unsigned int HEIGHT,
    unsigned int WIDTH,
    unsigned int DEPTH,
    unsigned int STRIDE_HEIGHT,
    unsigned int STRIDE_WIDTH,
    unsigned int STRIDE_DEPTH,
    unsigned int COARSE_OUT,
    typename T>
void fork_3d(hls::stream<T> &in, hls::stream<T> out[COARSE_OUT])
{

#pragma HLS INLINE OFF

    // set all parameters as constants
    const unsigned int batch = BATCH;
    const unsigned int height = HEIGHT;
    const unsigned int width = WIDTH;
    const unsigned int depth = DEPTH;
    const unsigned int strideh = STRIDE_HEIGHT;
    const unsigned int stridew = STRIDE_WIDTH;
    const unsigned int strided = STRIDE_DEPTH;
    const unsigned int channels = CHANNELS;
    const unsigned int coarse_out = COARSE_OUT;

    const unsigned int occurrence_coarse_distance = batch * height * width * depth * channels;

    T local_cache;

loop_batch:
    for (unsigned int batch_index = 0; batch_index < batch; batch_index++)
    {
    loop_rows:
        for (unsigned int row_index = 0; row_index < height; row_index++)
        {
        loop_cols:
            for (unsigned int col_index = 0; col_index < width; col_index++)
            {
            loop_depth:
                for (unsigned int depth_index = 0; depth_index < depth; depth_index++)
                {
                loop_channels:
                    for (unsigned int channel_index = 0; channel_index < channels; channel_index++)
                    {
#pragma HLS LOOP_FLATTEN
#pragma HLS PIPELINE II = 1 rewind
                    coarse_loop:
                        for (unsigned int coarse_index = 0; coarse_index < coarse_out; coarse_index++)
                        {
                            if (coarse_index == 0)
                            {
                                DO_PRAGMA(HLS OCCURRENCE cycle = occurrence_coarse_distance)
                                local_cache = in.read();
                            }
                            if ((row_index % strideh == 0) &&
                                (col_index % stridew == 0) &&
                                (depth_index % strided == 0))
                            {
                                out[coarse_index].write(local_cache);
                            }
                        }
                    }
                }
            }
        }
    }
}
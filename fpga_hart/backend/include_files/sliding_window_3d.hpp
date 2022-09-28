#pragma once

#include "common_.hpp"

template <
    unsigned int BATCH,
    unsigned int CHANNELS,
    unsigned int HEIGHT,
    unsigned int WIDTH,
    unsigned int DEPTH,
    unsigned int K_H,
    unsigned int K_W,
    unsigned int K_D,
    unsigned int PAD_H,
    unsigned int PAD_W,
    unsigned int PAD_D,
    typename T>
void sliding_window_3d_windows(
    hls::stream<T> &in,
    hls::stream<T> frame_buffer[K_H][K_W][K_D])
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
    const unsigned int padh = PAD_H;
    const unsigned int padw = PAD_W;
    const unsigned int padd = PAD_D;

    const unsigned int padded_height = height + 2 * padh;
    const unsigned int padded_width = width + 2 * padw;
    const unsigned int padded_depth = depth + 2 * padd;
    const unsigned int kh_minus = K_H - 1;
    const unsigned int kw_minus = K_W - 1;
    const unsigned int kd_minus = K_D - 1;
    const unsigned int semi_padded_height = height + padh - 1;
    const unsigned int semi_padded_width = width + padw - 1;
    const unsigned int semi_padded_depth = depth + padd - 1;

    hls::stream<T> line_buffer_3d[kh][kw_minus];
    DO_PRAGMA(HLS STREAM variable = line_buffer_3d depth = (channels * (padded_depth) + 1))
#pragma HLS ARRAY_PARTITION variable = line_buffer_3d complete dim = 0
#pragma HLS BIND_STORAGE variable = line_buffer_3d type = fifo impl = bram

    hls::stream<T> line_buffer_2d[kh_minus];
    DO_PRAGMA(HLS STREAM variable = line_buffer_2d depth = (channels * ((padded_depth) * (padded_width) - (kw_minus)*depth - (kd_minus)) + 1))
#pragma HLS ARRAY_PARTITION variable = line_buffer_2d complete dim = 0
#pragma HLS BIND_STORAGE variable = line_buffer_2d type = fifo impl = bram

    hls::stream<T> window_buffer_3d[kh][kw][kd_minus];
    DO_PRAGMA(HLS STREAM variable = window_buffer_3d depth = (channels + 1))
#pragma HLS ARRAY_PARTITION variable = window_buffer_3d complete dim = 0
#pragma HLS BIND_STORAGE variable = window_buffer_3d type = fifo impl = bram

    T cube_cache[kh][kw][kd];
#pragma HLS ARRAY_PARTITION variable = cube_cache complete dim = 0

in_loop_batch:
    for (unsigned int batch_index = 0; batch_index < batch; batch_index++)
    {
    in_loop_rows:
        for (unsigned int row_index = 0; row_index < padded_height; row_index++)
        {
        in_loop_cols:
            for (unsigned int col_index = 0; col_index < padded_width; col_index++)
            {
            in_loop_depth:
                for (unsigned int depth_index = 0; depth_index < padded_depth; depth_index++)
                {
                in_loop_channels:
                    for (unsigned int channel_index = 0; channel_index < channels; channel_index++)
                    {

#pragma HLS LOOP_FLATTEN
#pragma HLS PIPELINE II = 1
#pragma HLS DEPENDENCE variable = line_buffer_3d WAR intra true
#pragma HLS DEPENDENCE variable = line_buffer_2d WAR intra true
#pragma HLS DEPENDENCE variable = window_buffer_3d WAR intra true
#pragma HLS DEPENDENCE variable = cube_cache WAR intra true

                        T pixel;
                        // read in pixel
                        if (row_index < padh || row_index > semi_padded_height)
                        {
                            pixel = 0;
                        }
                        else if (col_index < padw || col_index > semi_padded_width)
                        {
                            pixel = 0;
                        }
                        else if (depth_index < padd || depth_index > semi_padded_depth)
                        {
                            pixel = 0;
                        }
                        else
                        {
                            pixel = in.read();
                        }

                        // init first part of window cache
                        if (row_index < kh_minus)
                        {
                            if (col_index <= kw_minus)
                            {
                                if (depth_index < kd_minus)
                                {
                                    window_buffer_3d[row_index][col_index][depth_index].write(pixel);
                                    cube_cache[row_index][col_index][depth_index] = pixel;
                                    if (col_index != 0)
                                    {
                                        line_buffer_3d[row_index][col_index - 1].write(pixel);
                                    }
                                }
                                else
                                {
                                    if (col_index != kw_minus)
                                    {
                                        line_buffer_3d[row_index][col_index].write(pixel);
                                    }
                                    else
                                    {
                                        line_buffer_2d[row_index].write(pixel);
                                    }
                                }
                            }
                            else
                            {
                                line_buffer_2d[row_index].write(pixel);
                            }
                        }
                        else if (row_index == kh_minus && col_index < kw_minus)
                        {
                            if (depth_index < kd_minus)
                            {
                                window_buffer_3d[row_index][col_index][depth_index].write(pixel);
                                cube_cache[row_index][col_index][depth_index] = pixel;
                                if (col_index != 0)
                                {
                                    line_buffer_3d[row_index][col_index - 1].write(pixel);
                                }
                            }
                            else
                            {
                                line_buffer_3d[row_index][col_index].write(pixel);
                            }
                        }
                        else if (row_index == kh_minus && col_index == kw_minus && depth_index < kd_minus)
                        {
                            window_buffer_3d[row_index][col_index][depth_index].write(pixel);
                            cube_cache[row_index][col_index][depth_index] = pixel;
                            line_buffer_3d[row_index][col_index - 1].write(pixel);
                        }
                        else
                        {
                            // read window buffer 3d into cube cache
                            for (unsigned char k1 = 0; k1 < kh; k1++)
                            {
                                if (k1 < kh_minus)
                                {
                                    T tmp_lb = line_buffer_2d[k1].read();
                                    cube_cache[k1][kw_minus][kd_minus] = tmp_lb;
                                }
                                for (unsigned char k2 = 0; k2 < kw; k2++)
                                {
                                    if (k2 < kw_minus)
                                    {
                                        T tmp_lb = line_buffer_3d[k1][k2].read();
                                        cube_cache[k1][k2][kd_minus] = tmp_lb;
                                    }
                                    for (unsigned char k3 = 0; k3 < kd_minus; k3++)
                                    {
                                        T tmp_wb = window_buffer_3d[k1][k2][k3].read();
                                        cube_cache[k1][k2][k3] = tmp_wb;
                                    }
                                }
                            }

                            cube_cache[kh_minus][kw_minus][kd_minus] = pixel;

                            // update window buffer 3d
                            if (!(
                                    (row_index == padded_height - 1) &&
                                    (col_index == padded_width - 1) &&
                                    (depth_index == padded_depth - 1)))
                            {
                                for (unsigned char k1 = 0; k1 < kh; k1++)
                                {
                                    for (unsigned char k2 = 0; k2 < kw; k2++)
                                    {
                                        for (unsigned char k3 = 0; k3 < kd_minus; k3++)
                                        {
                                            window_buffer_3d[k1][k2][k3].write(cube_cache[k1][k2][k3 + 1]);
                                        }
                                    }
                                }
                            }

                            // update the line buffer 3d
                            if (!(
                                    (row_index == padded_height - 1) &&
                                    (col_index == padded_width - 1)))
                            {
                                for (unsigned char k1 = 0; k1 < kh; k1++)
                                {
                                    for (unsigned char k2 = 0; k2 < kw_minus; k2++)
                                    {
                                        line_buffer_3d[k1][k2].write(cube_cache[k1][k2 + 1][kd_minus]);
                                    }
                                }
                            }

                            // update the line buffer 2d
                            if (!(
                                    (
                                        (row_index == padded_height - 1) &&
                                        (col_index == kw_minus) &&
                                        (depth_index >= kd_minus)) ||
                                    ((row_index == padded_height - 1) &&
                                     (col_index > kw_minus))))
                            {
                                for (unsigned char k1 = 0; k1 < kh_minus; k1++)
                                {
                                    line_buffer_2d[k1].write(cube_cache[k1 + 1][0][0]);
                                }
                            }

                            for (unsigned char k1 = 0; k1 < kh; k1++)
                            {
                                for (unsigned char k2 = 0; k2 < kw; k2++)
                                {
                                    for (unsigned char k3 = 0; k3 < kd; k3++)
                                    {
                                        frame_buffer[k1][k2][k3].write(cube_cache[k1][k2][k3]);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
#ifdef DEBUG_PRINTS
    for (int h = 0; h < kh; h++)
    {
        for (int w = 0; w < kw_minus; w++)
        {
            cout << line_buffer_3d[h][w].empty() << endl;
        }
    }
    cout << "********" << endl;
    for (int h = 0; h < kh; h++)
    {
        cout << line_buffer_2d[h].empty() << endl;
    }
    cout << "********" << endl;
    for (int h = 0; h < kh; h++)
    {
        for (int w = 0; w < kw; w++)
        {
            for (int d = 0; d < kd_minus; d++)
            {
                cout << window_buffer_3d[h][w][d].empty() << endl;
            }
        }
    }
#endif
}

template <
    unsigned int BATCH,
    unsigned int CHANNELS,
    unsigned int HEIGHT,
    unsigned int WIDTH,
    unsigned int DEPTH,
    unsigned int K_H,
    unsigned int K_W,
    unsigned int K_D,
    unsigned int PAD_H,
    unsigned int PAD_W,
    unsigned int PAD_D,
    unsigned int STRIDE_H,
    unsigned int STRIDE_W,
    unsigned int STRIDE_D,
    typename T>
void sliding_window_3d_filtered_windows(
    hls::stream<T> frame_buffer[K_H][K_W][K_D],
    hls::stream<T> out[K_H][K_W][K_D])
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
    const unsigned int padh = PAD_H;
    const unsigned int padw = PAD_W;
    const unsigned int padd = PAD_D;
    const unsigned int strideh = STRIDE_H;
    const unsigned int stridew = STRIDE_W;
    const unsigned int strided = STRIDE_D;

out_loop_batch:
    for (unsigned int batch_index = 0; batch_index < batch; batch_index++)
    {
    out_loop_rows:
        for (unsigned int row_index = 0; row_index < (height + 2 * padh); row_index++)
        {
        out_loop_cols:
            for (unsigned int col_index = 0; col_index < (width + 2 * padw); col_index++)
            {
            out_loop_depth:
                for (unsigned int depth_index = 0; depth_index < (depth + 2 * padd); depth_index++)
                {
                out_loop_channels:
                    for (unsigned int channel_index = 0; channel_index < channels; channel_index++)
                    {
#pragma HLS LOOP_FLATTEN
#pragma HLS PIPELINE II = 1 rewind
                        if (!(
                                (row_index < (kh - 1)) ||
                                (row_index == (kh - 1) && col_index < kw - 1) ||
                                (row_index == (kh - 1) && col_index == kw - 1 && depth_index < kd - 1)))
                        {
                            for (unsigned char k1 = 0; k1 < kh; k1++)
                            {
                                for (unsigned char k2 = 0; k2 < kw; k2++)
                                {
#ifdef DEBUG_PRINTS
                                    std::cout << "[";
#endif
                                    for (unsigned char k3 = 0; k3 < kd; k3++)
                                    {
                                        T tmp = frame_buffer[k1][k2][k3].read();
                                        if (
                                            (row_index >= (kh - 1)) &&
                                            ((row_index - kh + 1) % strideh == 0) &&
                                            (col_index >= (kw - 1)) &&
                                            ((col_index - kw + 1) % stridew == 0) &&
                                            (depth_index >= (kd - 1)) &&
                                            ((depth_index - kd + 1) % strided == 0))
                                        {
                                            out[k1][k2][k3].write(tmp);
#ifdef DEBUG_PRINTS
                                            std::cout << tmp.to_float() << " ";
#endif
                                        }
                                    }
#ifdef DEBUG_PRINTS
                                    std::cout << "]" << std::endl;
#endif
                                }
#ifdef DEBUG_PRINTS
                                std::cout << std::endl;
#endif
                            }
#ifdef DEBUG_PRINTS
                            std::cout << "************************" << std::endl;
#endif
                        }
                    }
                }
            }
        }
    }
}

template <
    unsigned int BATCH,
    unsigned int CHANNELS,
    unsigned int HEIGHT,
    unsigned int WIDTH,
    unsigned int DEPTH,
    unsigned int K_H,
    unsigned int K_W,
    unsigned int K_D,
    unsigned int PAD_H,
    unsigned int PAD_W,
    unsigned int PAD_D,
    unsigned int STRIDE_H,
    unsigned int STRIDE_W,
    unsigned int STRIDE_D,
    typename sw_t>
void sliding_window_3d(
    hls::stream<sw_t> &in,
    hls::stream<sw_t> out[K_H][K_W][K_D])
{

#pragma HLS INLINE OFF

#pragma HLS DATAFLOW

    const unsigned char kernel_size_x = K_H;
    const unsigned char kernel_size_y = K_W;
    const unsigned char kernel_size_z = K_D;

    hls::stream<sw_t> frame_buffer[kernel_size_x][kernel_size_y][kernel_size_z];
#pragma HLS STREAM variable = frame_buffer
#pragma HLS ARRAY_PARTITION variable = frame_buffer complete dim = 0
    //#pragma HLS BIND_STORAGE variable=frame_buffer type=fifo impl=bram

    sliding_window_3d_windows<
        BATCH,
        CHANNELS,
        HEIGHT,
        WIDTH,
        DEPTH,
        K_H,
        K_W,
        K_D,
        PAD_H,
        PAD_W,
        PAD_D,
        sw_t>(in, frame_buffer);

    sliding_window_3d_filtered_windows<
        BATCH,
        CHANNELS,
        HEIGHT,
        WIDTH,
        DEPTH,
        K_H,
        K_W,
        K_D,
        PAD_H,
        PAD_W,
        PAD_D,
        STRIDE_H,
        STRIDE_W,
        STRIDE_D,
        sw_t>(frame_buffer, out);
}

/* Sliding Window 3D with support for RUN time parameters for padding and stride */

template <
    unsigned int BATCH,
    unsigned int CHANNELS,
    unsigned int HEIGHT,
    unsigned int WIDTH,
    unsigned int DEPTH,
    unsigned int K_H,
    unsigned int K_W,
    unsigned int K_D,
    unsigned int PAD_H,
    unsigned int PAD_W,
    unsigned int PAD_D,
    typename T>
void sliding_window_3d_windows(
    hls::stream<T> &in,
    hls::stream<T> frame_buffer[K_H][K_W][K_D],
    unsigned int pad_h_rt,
    unsigned int pad_w_rt,
    unsigned int pad_d_rt)
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
    const unsigned int padh = PAD_H;
    const unsigned int padw = PAD_W;
    const unsigned int padd = PAD_D;

    const unsigned int padded_height = height + 2 * padh;
    const unsigned int padded_width = width + 2 * padw;
    const unsigned int padded_depth = depth + 2 * padd;
    const unsigned int kh_minus = K_H - 1;
    const unsigned int kw_minus = K_W - 1;
    const unsigned int kd_minus = K_D - 1;
    const unsigned int semi_padded_height = height + padh - 1;
    const unsigned int semi_padded_width = width + padw - 1;
    const unsigned int semi_padded_depth = depth + padd - 1;

    hls::stream<T> line_buffer_3d[kh][kw_minus];
    DO_PRAGMA(HLS STREAM variable = line_buffer_3d depth = (channels * (padded_depth) + 1))
#pragma HLS ARRAY_PARTITION variable = line_buffer_3d complete dim = 0
#pragma HLS BIND_STORAGE variable = line_buffer_3d type = fifo impl = bram

    hls::stream<T> line_buffer_2d[kh_minus];
    DO_PRAGMA(HLS STREAM variable = line_buffer_2d depth = (channels * ((padded_depth) * (padded_width) - (kw_minus)*depth - (kd_minus)) + 1))
#pragma HLS ARRAY_PARTITION variable = line_buffer_2d complete dim = 0
#pragma HLS BIND_STORAGE variable = line_buffer_2d type = fifo impl = bram

    hls::stream<T> window_buffer_3d[kh][kw][kd_minus];
    DO_PRAGMA(HLS STREAM variable = window_buffer_3d depth = (channels + 1))
#pragma HLS ARRAY_PARTITION variable = window_buffer_3d complete dim = 0
#pragma HLS BIND_STORAGE variable = window_buffer_3d type = fifo impl = bram

    T cube_cache[kh][kw][kd];
#pragma HLS ARRAY_PARTITION variable = cube_cache complete dim = 0

in_loop_batch:
    for (unsigned int batch_index = 0; batch_index < batch; batch_index++)
    {
    in_loop_rows:
        for (unsigned int row_index = 0; row_index < padded_height; row_index++)
        {
        in_loop_cols:
            for (unsigned int col_index = 0; col_index < padded_width; col_index++)
            {
            in_loop_depth:
                for (unsigned int depth_index = 0; depth_index < padded_depth; depth_index++)
                {
                in_loop_channels:
                    for (unsigned int channel_index = 0; channel_index < channels; channel_index++)
                    {

#pragma HLS LOOP_FLATTEN
#pragma HLS PIPELINE II = 1
#pragma HLS DEPENDENCE variable = line_buffer_3d WAR intra true
#pragma HLS DEPENDENCE variable = line_buffer_2d WAR intra true
#pragma HLS DEPENDENCE variable = window_buffer_3d WAR intra true
#pragma HLS DEPENDENCE variable = cube_cache WAR intra true

                        T pixel;
                        // read in pixel
                        if (row_index < padh || row_index > semi_padded_height)
                        {
                            pixel = 0;
                        }
                        else if (col_index < padw || col_index > semi_padded_width)
                        {
                            pixel = 0;
                        }
                        else if (depth_index < padd || depth_index > semi_padded_depth)
                        {
                            pixel = 0;
                        }
                        else
                        {
                            pixel = in.read();
                        }

                        // init first part of window cache
                        if (row_index < kh_minus)
                        {
                            if (col_index <= kw_minus)
                            {
                                if (depth_index < kd_minus)
                                {
                                    window_buffer_3d[row_index][col_index][depth_index].write(pixel);
                                    cube_cache[row_index][col_index][depth_index] = pixel;
                                    if (col_index != 0)
                                    {
                                        line_buffer_3d[row_index][col_index - 1].write(pixel);
                                    }
                                }
                                else
                                {
                                    if (col_index != kw_minus)
                                    {
                                        line_buffer_3d[row_index][col_index].write(pixel);
                                    }
                                    else
                                    {
                                        line_buffer_2d[row_index].write(pixel);
                                    }
                                }
                            }
                            else
                            {
                                line_buffer_2d[row_index].write(pixel);
                            }
                        }
                        else if (row_index == kh_minus && col_index < kw_minus)
                        {
                            if (depth_index < kd_minus)
                            {
                                window_buffer_3d[row_index][col_index][depth_index].write(pixel);
                                cube_cache[row_index][col_index][depth_index] = pixel;
                                if (col_index != 0)
                                {
                                    line_buffer_3d[row_index][col_index - 1].write(pixel);
                                }
                            }
                            else
                            {
                                line_buffer_3d[row_index][col_index].write(pixel);
                            }
                        }
                        else if (row_index == kh_minus && col_index == kw_minus && depth_index < kd_minus)
                        {
                            window_buffer_3d[row_index][col_index][depth_index].write(pixel);
                            cube_cache[row_index][col_index][depth_index] = pixel;
                            line_buffer_3d[row_index][col_index - 1].write(pixel);
                        }
                        else
                        {
                            // read window buffer 3d into cube cache
                            for (unsigned char k1 = 0; k1 < kh; k1++)
                            {
                                if (k1 < kh_minus)
                                {
                                    T tmp_lb = line_buffer_2d[k1].read();
                                    cube_cache[k1][kw_minus][kd_minus] = tmp_lb;
                                }
                                for (unsigned char k2 = 0; k2 < kw; k2++)
                                {
                                    if (k2 < kw_minus)
                                    {
                                        T tmp_lb = line_buffer_3d[k1][k2].read();
                                        cube_cache[k1][k2][kd_minus] = tmp_lb;
                                    }
                                    for (unsigned char k3 = 0; k3 < kd_minus; k3++)
                                    {
                                        T tmp_wb = window_buffer_3d[k1][k2][k3].read();
                                        cube_cache[k1][k2][k3] = tmp_wb;
                                    }
                                }
                            }

                            cube_cache[kh_minus][kw_minus][kd_minus] = pixel;

                            // update window buffer 3d
                            if (!(
                                    (row_index == padded_height - 1) &&
                                    (col_index == padded_width - 1) &&
                                    (depth_index == padded_depth - 1)))
                            {
                                for (unsigned char k1 = 0; k1 < kh; k1++)
                                {
                                    for (unsigned char k2 = 0; k2 < kw; k2++)
                                    {
                                        for (unsigned char k3 = 0; k3 < kd_minus; k3++)
                                        {
                                            window_buffer_3d[k1][k2][k3].write(cube_cache[k1][k2][k3 + 1]);
                                        }
                                    }
                                }
                            }

                            // update the line buffer 3d
                            if (!(
                                    (row_index == padded_height - 1) &&
                                    (col_index == padded_width - 1)))
                            {
                                for (unsigned char k1 = 0; k1 < kh; k1++)
                                {
                                    for (unsigned char k2 = 0; k2 < kw_minus; k2++)
                                    {
                                        line_buffer_3d[k1][k2].write(cube_cache[k1][k2 + 1][kd_minus]);
                                    }
                                }
                            }

                            // update the line buffer 2d
                            if (!(
                                    (
                                        (row_index == padded_height - 1) &&
                                        (col_index == kw_minus) &&
                                        (depth_index >= kd_minus)) ||
                                    ((row_index == padded_height - 1) &&
                                     (col_index > kw_minus))))
                            {
                                for (unsigned char k1 = 0; k1 < kh_minus; k1++)
                                {
                                    line_buffer_2d[k1].write(cube_cache[k1 + 1][0][0]);
                                }
                            }

                            for (unsigned char k1 = 0; k1 < kh; k1++)
                            {
                                for (unsigned char k2 = 0; k2 < kw; k2++)
                                {
                                    for (unsigned char k3 = 0; k3 < kd; k3++)
                                    {
                                        frame_buffer[k1][k2][k3].write(cube_cache[k1][k2][k3]);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
#ifdef DEBUG_PRINTS
    for (int h = 0; h < kh; h++)
    {
        for (int w = 0; w < kw_minus; w++)
        {
            cout << line_buffer_3d[h][w].empty() << endl;
        }
    }
    cout << "********" << endl;
    for (int h = 0; h < kh; h++)
    {
        cout << line_buffer_2d[h].empty() << endl;
    }
    cout << "********" << endl;
    for (int h = 0; h < kh; h++)
    {
        for (int w = 0; w < kw; w++)
        {
            for (int d = 0; d < kd_minus; d++)
            {
                cout << window_buffer_3d[h][w][d].empty() << endl;
            }
        }
    }
#endif
}

template <
    unsigned int BATCH,
    unsigned int CHANNELS,
    unsigned int HEIGHT,
    unsigned int WIDTH,
    unsigned int DEPTH,
    unsigned int K_H,
    unsigned int K_W,
    unsigned int K_D,
    unsigned int PAD_H,
    unsigned int PAD_W,
    unsigned int PAD_D,
    unsigned int STRIDE_H,
    unsigned int STRIDE_W,
    unsigned int STRIDE_D,
    typename T>
void sliding_window_3d_filtered_windows(
    hls::stream<T> frame_buffer[K_H][K_W][K_D],
    hls::stream<T> out[K_H][K_W][K_D],
    unsigned int pad_h_rt,
    unsigned int pad_w_rt,
    unsigned int pad_d_rt,
    unsigned int stride_h_rt,
    unsigned int stride_w_rt,
    unsigned int stride_d_rt)
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
    const unsigned int padh = PAD_H;
    const unsigned int padw = PAD_W;
    const unsigned int padd = PAD_D;
    const unsigned int strideh = STRIDE_H;
    const unsigned int stridew = STRIDE_W;
    const unsigned int strided = STRIDE_D;

out_loop_batch:
    for (unsigned int batch_index = 0; batch_index < batch; batch_index++)
    {
    out_loop_rows:
        for (unsigned int row_index = 0; row_index < (height + 2 * padh); row_index++)
        {
        out_loop_cols:
            for (unsigned int col_index = 0; col_index < (width + 2 * padw); col_index++)
            {
            out_loop_depth:
                for (unsigned int depth_index = 0; depth_index < (depth + 2 * padd); depth_index++)
                {
                out_loop_channels:
                    for (unsigned int channel_index = 0; channel_index < channels; channel_index++)
                    {
#pragma HLS LOOP_FLATTEN
#pragma HLS PIPELINE II = 1 rewind
                        if (!(
                                (row_index < (kh - 1)) ||
                                (row_index == (kh - 1) && col_index < kw - 1) ||
                                (row_index == (kh - 1) && col_index == kw - 1 && depth_index < kd - 1)))
                        {
                            for (unsigned char k1 = 0; k1 < kh; k1++)
                            {
                                for (unsigned char k2 = 0; k2 < kw; k2++)
                                {
#ifdef DEBUG_PRINTS
                                    std::cout << "[";
#endif
                                    for (unsigned char k3 = 0; k3 < kd; k3++)
                                    {
                                        T tmp = frame_buffer[k1][k2][k3].read();
                                        if (
                                            (row_index >= (kh - 1)) &&
                                            ((row_index - kh + 1) % strideh == 0) &&
                                            (col_index >= (kw - 1)) &&
                                            ((col_index - kw + 1) % stridew == 0) &&
                                            (depth_index >= (kd - 1)) &&
                                            ((depth_index - kd + 1) % strided == 0))
                                        {
                                            out[k1][k2][k3].write(tmp);
#ifdef DEBUG_PRINTS
                                            std::cout << tmp.to_float() << " ";
#endif
                                        }
                                    }
#ifdef DEBUG_PRINTS
                                    std::cout << "]" << std::endl;
#endif
                                }
#ifdef DEBUG_PRINTS
                                std::cout << std::endl;
#endif
                            }
#ifdef DEBUG_PRINTS
                            std::cout << "************************" << std::endl;
#endif
                        }
                    }
                }
            }
        }
    }
}

template <
    unsigned int BATCH,
    unsigned int CHANNELS,
    unsigned int HEIGHT,
    unsigned int WIDTH,
    unsigned int DEPTH,
    unsigned int K_H,
    unsigned int K_W,
    unsigned int K_D,
    unsigned int PAD_H,
    unsigned int PAD_W,
    unsigned int PAD_D,
    unsigned int STRIDE_H,
    unsigned int STRIDE_W,
    unsigned int STRIDE_D,
    typename sw_t>
void sliding_window_3d(
    hls::stream<sw_t> &in,
    hls::stream<sw_t> out[K_H][K_W][K_D],
    unsigned int pad_h_rt,
    unsigned int pad_w_rt,
    unsigned int pad_d_rt,
    unsigned int stride_h_rt,
    unsigned int stride_w_rt,
    unsigned int stride_d_rt)
{

#pragma HLS INLINE OFF

#pragma HLS DATAFLOW

    const unsigned char kernel_size_x = K_H;
    const unsigned char kernel_size_y = K_W;
    const unsigned char kernel_size_z = K_D;

    hls::stream<sw_t>
        frame_buffer[kernel_size_x][kernel_size_y][kernel_size_z];
#pragma HLS STREAM variable = frame_buffer
#pragma HLS ARRAY_PARTITION variable = frame_buffer complete dim = 0
    //#pragma HLS BIND_STORAGE variable=frame_buffer type=fifo impl=bram

    sliding_window_3d_windows<
        BATCH,
        CHANNELS,
        HEIGHT,
        WIDTH,
        DEPTH,
        K_H,
        K_W,
        K_D,
        PAD_H,
        PAD_W,
        PAD_D,
        sw_t>(in, frame_buffer, pad_h_rt, pad_w_rt, pad_d_rt);

    sliding_window_3d_filtered_windows<
        BATCH,
        CHANNELS,
        HEIGHT,
        WIDTH,
        DEPTH,
        K_H,
        K_W,
        K_D,
        PAD_H,
        PAD_W,
        PAD_D,
        STRIDE_H,
        STRIDE_W,
        STRIDE_D,
        sw_t>(frame_buffer, out, pad_h_rt, pad_w_rt, pad_d_rt, stride_h_rt, stride_w_rt, stride_d_rt);
}
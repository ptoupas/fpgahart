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
    const unsigned int batch        = BATCH;
    const unsigned int height       = HEIGHT;
    const unsigned int width        = WIDTH;
    const unsigned int depth        = DEPTH;
    const unsigned int channels     = CHANNELS;
	const unsigned int kh           = K_H;
	const unsigned int kw           = K_W;
	const unsigned int kd           = K_D;
	const unsigned int padh        = PAD_H;
	const unsigned int padw        = PAD_W;
	const unsigned int padd        = PAD_D;

	hls::stream<T> line_buffer_3d[kh][kw-1];
	DO_PRAGMA(HLS STREAM variable=line_buffer_3d depth=(channels*(depth + 2*padd) + 1))
#pragma HLS ARRAY_PARTITION variable=line_buffer_3d complete dim=0
#pragma HLS BIND_STORAGE variable=line_buffer_3d type=fifo impl=bram

	hls::stream<T> line_buffer_2d[kh-1];
	DO_PRAGMA(HLS STREAM variable=line_buffer_2d depth=(channels*((depth + 2*padd)*(width + 2*padw) - (kw - 1)*depth - (kd - 1)) + 1))
#pragma HLS ARRAY_PARTITION variable=line_buffer_2d complete dim=0
#pragma HLS BIND_STORAGE variable=line_buffer_2d type=fifo impl=bram

	hls::stream<T> window_buffer_3d[kh][kw][kd-1];
	DO_PRAGMA(HLS STREAM variable=window_buffer_3d depth=(channels + 1))
#pragma HLS ARRAY_PARTITION variable=window_buffer_3d complete dim=0
#pragma HLS BIND_STORAGE variable=window_buffer_3d type=fifo impl=bram

  T cube_cache[kh][kw][kd];
#pragma HLS ARRAY_PARTITION variable=cube_cache complete dim=0

  in_loop_batch: for(unsigned int batch_index=0;batch_index<batch;batch_index++) {
	  in_loop_rows: for(unsigned int row_index=0;row_index<height+2*padh;row_index++) {
		  in_loop_cols: for(unsigned int col_index=0;col_index<width+2*padw;col_index++) {
			  in_loop_depth: for(unsigned int depth_index=0;depth_index<depth+2*padd;depth_index++) {
				  in_loop_channels: for(unsigned int channel_index=0;channel_index<channels;channel_index++) {

#pragma HLS LOOP_FLATTEN
#pragma HLS PIPELINE II=1
#pragma HLS DEPENDENCE variable=line_buffer_3d WAR intra true
#pragma HLS DEPENDENCE variable=line_buffer_2d WAR intra true
#pragma HLS DEPENDENCE variable=window_buffer_3d WAR intra true
#pragma HLS DEPENDENCE variable=cube_cache WAR intra true

						T pixel;
						// read in pixel
						if( row_index < padh ) {
							pixel = 0;
						}
						else if ( row_index > height+padh-1 ) {
							pixel = 0;
						}
						else if ( col_index < padw ) {
							pixel = 0;
						}
						else if (col_index > width+padw-1 ) {
							pixel = 0;
						}
						else if ( depth_index < padd ) {
							pixel = 0;
						}
						else if (depth_index > depth+padd-1 ) {
							pixel = 0;
						}
						else {
							pixel = in.read();
						}

						// init first part of window cache
						if ( row_index < kh-1 ) {
							if ( col_index < kw-1 ) {
								if ( depth_index < kd-1 ) {
									window_buffer_3d[row_index][col_index][depth_index].write(pixel);
									cube_cache[row_index][col_index][depth_index] = pixel;
									if ( col_index > 0 ) {
										line_buffer_3d[row_index][col_index-1].write(pixel);
									}
								}
								else {
									line_buffer_3d[row_index][col_index].write(pixel);
								}
							}
							else if ( (col_index == kw-1 || col_index == kw) && depth_index < kd-1 ) {
								if (col_index == kw-1) {
									window_buffer_3d[row_index][col_index][depth_index].write(pixel);
									cube_cache[row_index][col_index][depth_index] = pixel;
									line_buffer_3d[row_index][col_index-1].write(pixel);
								}
								else if (col_index == kw) {
									line_buffer_2d[row_index].write(pixel);
								}
							}
							else {
								line_buffer_2d[row_index].write(pixel);
							}
						}
						else if (row_index == kh-1 && col_index < kw-1) {
							if (depth_index < kd-1) {
								window_buffer_3d[row_index][col_index][depth_index].write(pixel);
								cube_cache[row_index][col_index][depth_index] = pixel;
								if ( col_index > 0 ) {
									line_buffer_3d[row_index][col_index-1].write(pixel);
								}
							}
							else {
								line_buffer_3d[row_index][col_index].write(pixel);
							}
						}
						else if (row_index == kh-1 && col_index == kw-1 && depth_index < kd-1) {
							window_buffer_3d[row_index][col_index][depth_index].write(pixel);
							cube_cache[row_index][col_index][depth_index] = pixel;
							line_buffer_3d[row_index][col_index-1].write(pixel);
						}
						else {

							// read window buffer 3d into cube cache
							for(unsigned char k1=0;k1<kh;k1++) {
								for(unsigned char k2=0;k2<kw;k2++) {
									for(unsigned char k3=0;k3<kd-1;k3++) {
										T tmp_wb = window_buffer_3d[k1][k2][k3].read();
										cube_cache[k1][k2][k3] = tmp_wb;
									}
								}
							}

							// read out line buffer 3d to cube cache
							for(unsigned char k1=0;k1<kh;k1++) {
								for(unsigned char k2=0;k2<kw-1;k2++) {
									T tmp_lb = line_buffer_3d[k1][k2].read();
									cube_cache[k1][k2][kd-1] = tmp_lb;
								}
							}

							// read out line buffer 2d to cube cache
							for(unsigned char k1=0;k1<kh-1;k1++) {
								T tmp_lb = line_buffer_2d[k1].read();
								cube_cache[k1][kw-1][kd-1] = tmp_lb;
							}

							cube_cache[kh-1][kw-1][kd-1] = pixel;

							// update window buffer 3d
							if ( !(
									(row_index == height+2*padh-1) &&
									(col_index == width+2*padw-1) &&
									(depth_index == depth+2*padd-1)
							) ) {
								for(unsigned char k1=0;k1<kh;k1++) {
									for(unsigned char k2=0;k2<kw;k2++) {
										for(unsigned char k3=0;k3<kd-1;k3++) {
											window_buffer_3d[k1][k2][k3].write(cube_cache[k1][k2][k3+1]);
										}
									}
								}
							}

							// update the line buffer 3d
							if ( !(
									(row_index == height+2*padh-1) &&
									(col_index == width+2*padw-1)
							) ) {
								for(unsigned char k1=0;k1<kh;k1++) {
									for(unsigned char k2=0;k2<kw-1;k2++) {
										line_buffer_3d[k1][k2].write(cube_cache[k1][k2+1][kd-1]);
									}
								}
							}

							// update the line buffer 2d
							if ( !(
									(
									  (row_index == height+2*padh-1) &&
									  (col_index == kw-1) &&
									  (depth_index >= kd-1)
									) ||
									(
									  (row_index == height+2*padh-1) &&
									  (col_index > kw-1)
									)
							) ) {
								for(unsigned char k1=0;k1<kh-1;k1++) {
									line_buffer_2d[k1].write(cube_cache[k1+1][0][0]);
								}
							}

							for(unsigned char k1=0;k1<kh;k1++) {
								for(unsigned char k2=0;k2<kw;k2++) {
									for(unsigned char k3=0;k3<kd;k3++) {
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
  for (int h=0; h < kh; h++){
  	for (int w=0; w < kw-1; w++){
  		cout << line_buffer_3d[h][w].empty() << endl;
  	}
  }
  cout << "********" << endl;
  for (int h=0; h < kh; h++){
	cout << line_buffer_2d[h].empty() << endl;
  }
  cout << "********" << endl;
  for (int h=0; h < kh; h++){
  	for (int w=0; w < kw; w++){
	  for (int d=0; d < kd-1; d++){
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
    const unsigned int batch        = BATCH;
    const unsigned int height       = HEIGHT;
    const unsigned int width        = WIDTH;
    const unsigned int depth        = DEPTH;
    const unsigned int channels     = CHANNELS;
	const unsigned int kh           = K_H;
	const unsigned int kw           = K_W;
	const unsigned int kd           = K_D;
	const unsigned int padh        = PAD_H;
	const unsigned int padw        = PAD_W;
	const unsigned int padd        = PAD_D;
	const unsigned int strideh     = STRIDE_H;
	const unsigned int stridew     = STRIDE_W;
	const unsigned int strided     = STRIDE_D;

  out_loop_batch: for(unsigned int batch_index=0;batch_index<batch;batch_index++) {
		out_loop_rows: for(unsigned int row_index=0;row_index<(height+2*padh);row_index++) {
			out_loop_cols: for(unsigned int col_index=0;col_index<(width+2*padw);col_index++) {
				out_loop_depth: for(unsigned int depth_index=0;depth_index<(depth+2*padd);depth_index++) {
					out_loop_channels: for(unsigned int channel_index=0;channel_index<channels;channel_index++) {
#pragma HLS LOOP_FLATTEN
#pragma HLS PIPELINE II=1 rewind
						if ( !(
								( row_index < (kh-1)) ||
								( row_index == (kh-1) && col_index < kw-1 ) ||
								( row_index == (kh-1) && col_index == kw-1 && depth_index < kd-1)
							)  ) {
							for(unsigned char k1=0;k1<kh;k1++) {
								for(unsigned char k2=0;k2<kw;k2++) {
#ifdef DEBUG_PRINTS
									std::cout << "[";
#endif
									for(unsigned char k3=0;k3<kd;k3++) {
										T tmp = frame_buffer[k1][k2][k3].read();
										if (
												(row_index >= (kh-1)) &&
												((row_index-kh+1)%strideh == 0) &&
												(col_index >= (kw-1)) &&
												((col_index-kw+1)%stridew == 0) &&
												(depth_index >= (kd-1)) &&
												((depth_index-kd+1)%strided == 0)
											 ) {
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
		hls::stream<sw_t>& in,
		hls::stream<sw_t> out[K_H][K_W][K_D])
{

#pragma HLS INLINE OFF

#pragma HLS DATAFLOW

    const unsigned char kernel_size_x = K_H;
    const unsigned char kernel_size_y = K_W;
    const unsigned char kernel_size_z = K_D;

    hls::stream<sw_t> frame_buffer[kernel_size_x][kernel_size_y][kernel_size_z];
#pragma HLS STREAM variable=frame_buffer
#pragma HLS ARRAY_PARTITION variable=frame_buffer complete dim=0
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
						sw_t>
    (in,frame_buffer);

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
						sw_t>
    (frame_buffer,out);

}

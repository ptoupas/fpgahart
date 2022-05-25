#ifndef COMMON_TB_HPP_
#define COMMON_TB_HPP_

#define MALLOC_USAGE
#define ERROR_TOLERANCE 1e-1

#include "hls_stream.h"
#include "hls_math.h"
#include "ap_fixed.h"
#include <string>
#include <iostream>
#include <fstream>
#include <typeinfo>
#include <stdio.h>
#include <stdlib.h>

bool check_empty(ifstream &pFile)
{
	return pFile.peek() == ifstream::traits_type::eof();
}

template <typename T>
int checkStreamEqual(hls::stream<T> &res, hls::stream<T> &gt)
{
	int err = 0;
	while (!gt.empty())
	{
		if (res.empty())
		{
			printf("ERROR: empty early\n");
			return 1;
		}
		T tmp = res.read();
		T tmp_valid = gt.read();

		if (false)
			printf("%f,%f\n", tmp.to_float(), tmp_valid.to_float());

		if (
			(tmp.to_float() > tmp_valid.to_float() + ERROR_TOLERANCE) ||
			(tmp.to_float() < tmp_valid.to_float() - ERROR_TOLERANCE))
		{
			printf("ERROR: wrong value %f, %f, %f\n", tmp.to_float(), tmp_valid.to_float(), tmp.to_float() - tmp_valid.to_float());
			return 1;
			err++;
		}
	}

	if (!res.empty())
	{
		printf("ERROR: still data in stream\n");
		return 1;
		err++;
	}
	return err;
}

template <int SIZE, int STREAMS, typename T>
int get_index(int size, int steam)
{
	int index = (STREAMS * size) + steam;

	return index;
}

template <int SIZE, int STREAMS, int K_H, int K_W, int K_D, typename T>
int get_index(int size, int steam, int k1, int k2, int k3)
{
	int index = (STREAMS * K_H * K_W * K_D * size) +
				(K_H * K_W * K_D * steam) +
				(K_W * K_D * k1) +
				(K_D * k2) +
				k3;

	return index;
}

template <int SIZE, int STREAMS, typename T>
#ifdef MALLOC_USAGE
void load_data(string file_name, T *array)
#else
void load_data(string file_name, T array[SIZE][STREAMS])
#endif
{
	float f;
	ifstream input(file_name, ios::binary);
	for (int i = 0; i < SIZE; i++)
	{
		for (int j = 0; j < STREAMS; j++)
		{
			input.read(reinterpret_cast<char *>(&f), sizeof(float));
#ifdef MALLOC_USAGE
			array[get_index<SIZE, STREAMS, T>(i, j)] = T(f);
#else
			array[i][j] = T(f);
#endif
		}
	}
	if (!check_empty(input))
		cout << "Input data was not read correctly!" << endl;
	return;
}

template <int SIZE, int STREAMS, typename T>
#ifdef MALLOC_USAGE
void to_stream(T *array, hls::stream<T> out[STREAMS])
#else
void to_stream(T array[SIZE][STREAMS], hls::stream<T> out[STREAMS])
#endif
{
	std::cout << "Size: " << SIZE << ", Streams: " << STREAMS << std::endl;
	for (int i = 0; i < SIZE; i++)
	{
		for (int j = 0; j < STREAMS; j++)
		{
#ifdef MALLOC_USAGE
			out[j].write(array[get_index<SIZE, STREAMS, T>(i, j)]);
#else
			out[j].write(array[i][j]);
#endif
		}
	}
	return;
}

template <int SIZE, int STREAMS, int K_H, int K_W, int K_D, typename T>
#ifdef MALLOC_USAGE
void load_data(string file_name, T *array)
#else
void load_data(string file_name, T array[SIZE][STREAMS][K_H][K_W][K_D])
#endif
{
	float f;
	ifstream input(file_name, ios::binary);
	for (int i = 0; i < SIZE; i++)
	{
		for (int j = 0; j < STREAMS; j++)
		{
			for (int k1 = 0; k1 < K_H; k1++)
			{
				for (int k2 = 0; k2 < K_W; k2++)
				{
					for (int k3 = 0; k3 < K_D; k3++)
					{
						input.read(reinterpret_cast<char *>(&f), sizeof(float));
#ifdef MALLOC_USAGE
						array[get_index<SIZE, STREAMS, K_H, K_W, K_D, T>(i, j, k1, k2, k3)] = T(f);
#else
						array[i][j][k1][k2][k3] = T(f);
#endif
					}
				}
			}
		}
	}
	if (!check_empty(input))
		cout << "Input data was not read correctly!" << endl;
	return;
}

template <int SIZE, int STREAMS, int K_H, int K_W, int K_D, typename T>
#ifdef MALLOC_USAGE
void to_stream(T *array, hls::stream<T> out[STREAMS][K_H][K_W][K_D])
#else
void to_stream(T array[SIZE][STREAMS][K_H][K_W][K_D], hls::stream<T> out[STREAMS][K_H][K_W][K_D])
#endif
{
	std::cout << "Size: " << SIZE << ", Streams: " << STREAMS << ", K_H: " << K_H << ", K_W: " << K_W << ", K_D: " << K_D << std::endl;
	for (int i = 0; i < SIZE; i++)
	{
		for (int j = 0; j < STREAMS; j++)
		{
			for (int k1 = 0; k1 < K_H; k1++)
			{
				for (int k2 = 0; k2 < K_W; k2++)
				{
					for (int k3 = 0; k3 < K_D; k3++)
					{
#ifdef MALLOC_USAGE
						out[j][k1][k2][k3].write(array[get_index<SIZE, STREAMS, K_H, K_W, K_D, T>(i, j, k1, k2, k3)]);
#else
						out[j][k1][k2][k3].write(array[i][j][k1][k2][k3]);
#endif
					}
				}
			}
		}
	}
	return;
}

#endif // COMMON_TB_HPP_

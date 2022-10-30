#pragma once

#include "hls_stream.h"
#include "hls_math.h"
#include "ap_fixed.h"
#include "ap_axi_sdata.h"
#include <iostream>
#include <fstream>

using namespace std;

#define PRAGMA_SUB(x) _Pragma (#x)
#define DO_PRAGMA(x) PRAGMA_SUB(x)

#define DIVIDE(a,b) ((int) ((a)/(b)))

#define MIN(a,b) (((a)<(b))?(a):(b))

#define stream_t(x) hls::stream<x>

#define GET_OUT_DIM(d,p,k,s) ((int) (((d + 2*p -k) / s) + 1))

#define MALLOC_USAGE
//#define DEBUG_PRINTS
//#define FLOAT_USAGE

#define ERROR_TOLERANCE 1e-1

#define DATA_WIDTH   	16
#define INTEGER_BITS 	8

typedef ap_axis<DATA_WIDTH, 1, 1, 1> axi_stream_t;

#ifdef FLOAT_USAGE
typedef float datamath_t;
typedef float data_t;
typedef float accum_data_t;
#else
typedef ap_fixed<DATA_WIDTH,INTEGER_BITS> datamath_t;
typedef ap_fixed<DATA_WIDTH,INTEGER_BITS,AP_RND> data_t;
typedef ap_fixed<30,15,AP_RND> accum_data_t;
#endif

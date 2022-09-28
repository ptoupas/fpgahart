#pragma once

#include "common_.hpp"

/**
 *  FULLY CONNECTED FUNCTION (GEMM)
 */
template <
    unsigned int BATCH,
    unsigned int IN_FEATURES,
    unsigned int OUT_FEATURES,
    typename gemm_t>
void gemm(
    stream_t(gemm_t) & in,
    stream_t(gemm_t) & weights,
    stream_t(gemm_t) & out)
{

#pragma HLS INLINE OFF

    // set all parameters as constants
    const unsigned int batch = BATCH;
    const unsigned int in_features = IN_FEATURES;
    const unsigned int out_features = OUT_FEATURES;

    gemm_t res_buffer[out_features];
#pragma HLS ARRAY_PARTITION variable = res_buffer complete dim = 0
    gemm_t tmp_in, tmp_weight;
    for (unsigned long batch_idx = 0; batch_idx < batch; batch_idx++)
    {
        for (unsigned long feat_in = 0; feat_in < in_features; feat_in++)
        {
#pragma HLS PIPELINE II = 1 rewind
            tmp_in = in.read();
            for (unsigned long feat_out = 0; feat_out < out_features; feat_out++)
            {
                tmp_weight = weights.read();
                gemm_t prev = (feat_in == 0) ? gemm_t(0) : res_buffer[feat_out];
                res_buffer[feat_out] = prev + tmp_weight * tmp_in;
                if (feat_in == in_features - 1)
                {
                    out.write(res_buffer[feat_out]);
                }
            }
        }
    }
}

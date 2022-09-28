import os

from .codegen import *


def generate_gemm_cpp(name, config, partition_name):
    batch_size = 1
    in_features = config["shape_in"]
    out_features = config["shape_out"]
    weights = config["weights"]
    coarse_in_factor = config["coarse_in_factor"]
    coarse_out_factor = config["coarse_out_factor"]

    layer_name_lower = name.lower()
    layer_name_upper = name.upper()

    cpp = CppFile(
        os.path.join(
            os.getcwd(), "generated_files", partition_name, f"{layer_name_lower}.cpp"
        )
    )

    cpp(f'#include "{layer_name_lower}.hpp"', newlines=2)

    with cpp.block(
        f"void {layer_name_lower}_layer(\n\
        stream_t({layer_name_lower}_data_t) in[{layer_name_upper}_COARSE_IN],\n\
        stream_t({layer_name_lower}_data_t) weights[{layer_name_upper}_COARSE_OUT],\n\
        stream_t({layer_name_lower}_data_t) out[{layer_name_upper}_COARSE_OUT])"
    ):

        cpp("#pragma HLS INLINE OFF")
        cpp("#pragma HLS DATAFLOW", newlines=2)

        cpp("#pragma HLS ARRAY_PARTITION variable=in  complete dim=0")
        cpp("#pragma HLS ARRAY_PARTITION variable=weights  complete dim=0")
        cpp("#pragma HLS ARRAY_PARTITION variable=out complete dim=0", newlines=2)

        cpp(
            f"hls::stream<{layer_name_lower}_data_t> fork_in[{layer_name_upper}_COARSE_OUT];"
        )
        cpp("#pragma HLS STREAM variable=fork_in")
        cpp("#pragma HLS ARRAY_PARTITION variable=fork_in  complete dim=0", newlines=2)

        cpp(
            f"fork_3d<\n\
                {layer_name_upper}_FORK_BATCH_SIZE,\n\
                {layer_name_upper}_FORK_IN_FEATURES,\n\
                {layer_name_upper}_FORK_COARSE_OUT,\n\
                {layer_name_lower}_data_t\n\
            >(in[0],fork_in);",
            newlines=2,
        )

        with cpp.block(
            f"for(int coarseOutdex=0; coarseOutdex<{layer_name_upper}_COARSE_OUT; coarseOutdex++)"
        ):
            cpp("#pragma HLS unroll", newlines=2)

            cpp(
                f"gemm<\n\
                {layer_name_upper}_GEMM_BATCH_SIZE,\n\
                {layer_name_upper}_GEMM_IN_FEATURES,\n\
                {layer_name_upper}_GEMM_OUT_FEATURES,\n\
                {layer_name_lower}_data_t\n\
            >(fork_in[coarseOutdex],weights[coarseOutdex],out[coarseOutdex]);",
                newlines=2,
            )

    cpp.close()


def generate_gemm_hpp(name, config, partition_name):
    batch_size = 1
    in_features = config["shape_in"]
    out_features = config["shape_out"]
    weights = config["weights"]
    coarse_in_factor = config["coarse_in_factor"]
    coarse_out_factor = config["coarse_out_factor"]

    layer_name_lower = name.lower()
    layer_name_upper = name.upper()

    hpp = CppFile(
        os.path.join(
            os.getcwd(), "generated_files", partition_name, f"{layer_name_lower}.hpp"
        )
    )

    hpp("#pragma once", newlines=2)
    hpp('#include "common_.hpp"')
    hpp('#include "gemm_.hpp"')
    hpp('#include "fork_3d_.hpp"', newlines=2)

    hpp(f"#define {layer_name_upper}_BATCH_SIZE {batch_size}")
    hpp(f"#define {layer_name_upper}_IN_FEATURES {in_features}")
    hpp(f"#define {layer_name_upper}_OUT_FEATURES {out_features}")
    hpp(f"#define {layer_name_upper}_COARSE_IN {coarse_in_factor}")
    hpp(f"#define {layer_name_upper}_COARSE_OUT {coarse_out_factor}", newlines=2)

    hpp(f"#define {layer_name_upper}_FORK_BATCH_SIZE \t{layer_name_upper}_BATCH_SIZE")
    hpp(f"#define {layer_name_upper}_FORK_IN_FEATURES \t{layer_name_upper}_IN_FEATURES")
    hpp(
        f"#define {layer_name_upper}_FORK_OUT_FEATURES \tDIVIDE({layer_name_upper}_OUT_FEATURES, {layer_name_upper}_COARSE_OUT)"
    )
    hpp(f"#define {layer_name_upper}_FORK_COARSE_OUT \t{layer_name_upper}_COARSE_OUT")

    hpp(f"#define {layer_name_upper}_GEMM_BATCH_SIZE \t{layer_name_upper}_BATCH_SIZE")
    hpp(f"#define {layer_name_upper}_GEMM_IN_FEATURES \t{layer_name_upper}_IN_FEATURES")
    hpp(
        f"#define {layer_name_upper}_GEMM_OUT_FEATURES \tDIVIDE({layer_name_upper}_OUT_FEATURES, {layer_name_upper}_COARSE_OUT)"
    )

    hpp(
        f"typedef ap_fixed<16,8,AP_RND, AP_SAT> \t{layer_name_lower}_data_t;",
        newlines=3,
    )

    hpp(
        f"void {layer_name_lower}_layer(\n\
        stream_t({layer_name_lower}_data_t) in[{layer_name_upper}_COARSE_IN],\n\
        stream_t({layer_name_lower}_data_t) weights[{layer_name_upper}_COARSE_OUT],\n\
        stream_t({layer_name_lower}_data_t) out[{layer_name_upper}_COARSE_OUT]);"
    )

    hpp.close()


def generate_gemm_files(name, config, partition_name):
    if not os.path.exists(os.path.join(os.getcwd(), "generated_files", partition_name)):
        os.makedirs(os.path.join(os.getcwd(), "generated_files", partition_name))

    generate_gemm_hpp(name, config, partition_name)
    generate_gemm_cpp(name, config, partition_name)

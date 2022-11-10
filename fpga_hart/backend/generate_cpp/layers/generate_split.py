import os

from .codegen import *


def generate_split_cpp(name: str, config: dict, model_name: str, partition_name: str):
    batch_size = config['batch_size']
    channels = config['channels_out']
    depth = config['depth_out']
    height = config['height_out']
    width = config['width_out']
    depthwise = config['depthwise'] if 'depthwise' in config.keys() else 0
    coarse_factor = config['coarse_factor'] if 'coarse_factor' in config.keys() else config['coarse_out_factor']

    layer_name_lower = name.replace("GlobalAveragePool", "GAP").lower()
    layer_name_upper = name.replace("GlobalAveragePool", "GAP").upper()

    cpp = CppFile(os.path.join(os.getcwd(), "generated_files", model_name, partition_name, name, "src", f"{layer_name_lower}.cpp"))

    cpp(f"#include \"{layer_name_lower}.hpp\"", newlines=2)

    with cpp.block(f"void {layer_name_lower}_layer(\n\
        stream_t({layer_name_lower}_data_t) in[{layer_name_upper}_COARSE],\n\
        stream_t({layer_name_lower}_data_t) out_1[{layer_name_upper}_COARSE],\n\
        stream_t({layer_name_lower}_data_t) out_2[{layer_name_upper}_COARSE])"):

        cpp("#pragma HLS INLINE OFF")
        cpp("#pragma HLS DATAFLOW", newlines=2)

        cpp("#pragma HLS ARRAY_PARTITION variable=in  complete dim=0")
        cpp("#pragma HLS ARRAY_PARTITION variable=out_1 complete dim=0")
        cpp("#pragma HLS ARRAY_PARTITION variable=out_2 complete dim=0", newlines=2)

        with cpp.block(f"for(int coarseIndex=0; coarseIndex<{layer_name_upper}_COARSE; coarseIndex++)"):
            cpp("#pragma HLS unroll", newlines=2)

            cpp(f"split_3d<\n\
                {layer_name_upper}_SPLIT_BATCH_SIZE,\n\
                {layer_name_upper}_SPLIT_CHANNELS,\n\
                {layer_name_upper}_SPLIT_HEIGHT,\n\
                {layer_name_upper}_SPLIT_WIDTH,\n\
                {layer_name_upper}_SPLIT_DEPTH,\n\
                {layer_name_lower}_data_t\n\
            >(in[coarseIndex],out_1[coarseIndex],out_2[coarseIndex]);", newlines=2)

    cpp.close()

def generate_split_hpp(name: str, config: dict, model_name: str, partition_name: str):
    batch_size = config['batch_size']
    channels = config['channels_out']
    depth = config['depth_out']
    height = config['height_out']
    width = config['width_out']
    depthwise = config['depthwise'] if 'depthwise' in config.keys() else 0
    coarse_factor = config['coarse_factor'] if 'coarse_factor' in config.keys() else config['coarse_out_factor']

    layer_name_lower = name.replace("GlobalAveragePool", "GAP").lower()
    layer_name_upper = name.replace("GlobalAveragePool", "GAP").upper()

    hpp = CppFile(os.path.join(os.getcwd(), "generated_files", model_name, partition_name, name, "src", f"{layer_name_lower}.hpp"))

    hpp("#pragma once", newlines=2)
    hpp("#include \"common_.hpp\"")
    hpp("#include \"split_3d_.hpp\"", newlines=2)

    hpp(f"#define {layer_name_upper}_BATCH_SIZE {batch_size}")
    hpp(f"#define {layer_name_upper}_CHANNELS {channels}")
    hpp(f"#define {layer_name_upper}_DEPTH {depth}")
    hpp(f"#define {layer_name_upper}_HEIGHT {height}")
    hpp(f"#define {layer_name_upper}_WIDTH {width}", newlines=2)

    hpp(f"#define {layer_name_upper}_COARSE {coarse_factor}", newlines=2)

    hpp(f"#define {layer_name_upper}_SPLIT_BATCH_SIZE \t{layer_name_upper}_BATCH_SIZE")
    hpp(f"#define {layer_name_upper}_SPLIT_CHANNELS \tDIVIDE({layer_name_upper}_CHANNELS, {layer_name_upper}_COARSE)")
    hpp(f"#define {layer_name_upper}_SPLIT_DEPTH \t{layer_name_upper}_DEPTH")
    hpp(f"#define {layer_name_upper}_SPLIT_HEIGHT \t{layer_name_upper}_HEIGHT")
    hpp(f"#define {layer_name_upper}_SPLIT_WIDTH \t{layer_name_upper}_WIDTH", newlines=2)

    hpp(f"typedef ap_fixed<16,8,AP_RND, AP_SAT> \t{layer_name_lower}_data_t;", newlines=3)

    hpp(f"void {layer_name_lower}_layer(\n\
        stream_t({layer_name_lower}_data_t) in[{layer_name_upper}_COARSE],\n\
        stream_t({layer_name_lower}_data_t) out_1[{layer_name_upper}_COARSE],\n\
        stream_t({layer_name_lower}_data_t) out_2[{layer_name_upper}_COARSE]);")

    hpp.close()

def generate_split_files(name: str, config: dict, model_name: str, partition_name: str = ''):
    if not os.path.exists(os.path.join(os.getcwd(), "generated_files", model_name, partition_name, name, "src")):
        os.makedirs(os.path.join(os.getcwd(), "generated_files", model_name, partition_name, name, "src"))

    generate_split_hpp(name, config, model_name, partition_name)
    generate_split_cpp(name, config, model_name, partition_name)

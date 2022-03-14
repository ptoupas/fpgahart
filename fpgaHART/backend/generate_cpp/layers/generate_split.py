from .codegen import *
import os

def generate_split_cpp(name, config, partition_name):
    batch_size = config['shape_in'][0]
    channels = config['shape_in'][1]
    depth = config['shape_in'][2]
    height = config['shape_in'][3]
    width = config['shape_in'][4]
    coarse_factor = config['coarse_factor'] if 'coarse_factor' in config.keys() else config['coarse_out_factor']

    layer_name_lower = name.replace("GlobalAveragePool", "GAP").lower()
    layer_name_upper = name.replace("GlobalAveragePool", "GAP").upper()

    cpp = CppFile(os.path.join(os.getcwd(), "generated_files", partition_name, f"split_{layer_name_lower}.cpp"))

    cpp(f"#include \"split_{layer_name_lower}.hpp\"", newlines=2)

    with cpp.block(f"void split_{layer_name_lower}_layer(\n\
        stream_t(split_{layer_name_lower}_data_t) in[SPLIT_{layer_name_upper}_COARSE],\n\
        stream_t(split_{layer_name_lower}_data_t) out_1[SPLIT_{layer_name_upper}_COARSE],\n\
        stream_t(split_{layer_name_lower}_data_t) out_2[SPLIT_{layer_name_upper}_COARSE])"):

        cpp("#pragma HLS INLINE OFF")
        cpp("#pragma HLS DATAFLOW", newlines=2)

        cpp("#pragma HLS ARRAY_PARTITION variable=in  complete dim=0")
        cpp("#pragma HLS ARRAY_PARTITION variable=out_1 complete dim=0")
        cpp("#pragma HLS ARRAY_PARTITION variable=out_2 complete dim=0", newlines=2)

        with cpp.block(f"for(int coarseIndex=0; coarseIndex<SPLIT_{layer_name_upper}_COARSE; coarseIndex++)"):
            cpp("#pragma HLS unroll", newlines=2)

            cpp(f"split_3d<\n\
                SPLIT_{layer_name_upper}_RELU_BATCH_SIZE,\n\
                SPLIT_{layer_name_upper}_RELU_CHANNELS,\n\
                SPLIT_{layer_name_upper}_RELU_HEIGHT,\n\
                SPLIT_{layer_name_upper}_RELU_WIDTH,\n\
                SPLIT_{layer_name_upper}_RELU_DEPTH,\n\
                split_{layer_name_lower}_data_t\n\
            >(in[coarseIndex],out_1[coarseIndex],out_2[coarseIndex]);", newlines=2)
    
    cpp.close()

def generate_split_hpp(name, config, partition_name):
    batch_size = config['shape_in'][0]
    channels = config['shape_in'][1]
    depth = config['shape_in'][2]
    height = config['shape_in'][3]
    width = config['shape_in'][4]
    coarse_factor = config['coarse_factor'] if 'coarse_factor' in config.keys() else config['coarse_out_factor']

    layer_name_lower = name.replace("GlobalAveragePool", "GAP").lower()
    layer_name_upper = name.replace("GlobalAveragePool", "GAP").upper()

    hpp = CppFile(os.path.join(os.getcwd(), "generated_files", partition_name, f"split_{layer_name_lower}.hpp"))
    
    hpp("#pragma once", newlines=2)
    hpp("#include \"common_.hpp\"")
    hpp("#include \"split_3d_.hpp\"", newlines=2)

    hpp(f"#define SPLIT_{layer_name_upper}_BATCH_SIZE {batch_size}")
    hpp(f"#define SPLIT_{layer_name_upper}_CHANNELS {channels}")
    hpp(f"#define SPLIT_{layer_name_upper}_DEPTH {depth}")
    hpp(f"#define SPLIT_{layer_name_upper}_HEIGHT {height}")
    hpp(f"#define SPLIT_{layer_name_upper}_WIDTH {width}", newlines=2)

    hpp(f"#define SPLIT_{layer_name_upper}_COARSE {coarse_factor}", newlines=2)
    
    hpp(f"#define SPLIT_{layer_name_upper}_RELU_BATCH_SIZE \tSPLIT_{layer_name_upper}_BATCH_SIZE")
    hpp(f"#define SPLIT_{layer_name_upper}_RELU_CHANNELS \tDIVIDE(SPLIT_{layer_name_upper}_CHANNELS, SPLIT_{layer_name_upper}_COARSE)")
    hpp(f"#define SPLIT_{layer_name_upper}_RELU_DEPTH \tSPLIT_{layer_name_upper}_DEPTH")
    hpp(f"#define SPLIT_{layer_name_upper}_RELU_HEIGHT \tSPLIT_{layer_name_upper}_HEIGHT")
    hpp(f"#define SPLIT_{layer_name_upper}_RELU_WIDTH \tSPLIT_{layer_name_upper}_WIDTH", newlines=2)

    hpp(f"typedef ap_fixed<16,8,AP_RND, AP_SAT> \tsplit_{layer_name_lower}_data_t;", newlines=3)

    hpp(f"void split_{layer_name_lower}_layer(\n\
        stream_t(split_{layer_name_lower}_data_t) in[SPLIT_{layer_name_upper}_COARSE],\n\
        stream_t(split_{layer_name_lower}_data_t) out_1[SPLIT_{layer_name_upper}_COARSE],\n\
        stream_t(split_{layer_name_lower}_data_t) out_2[SPLIT_{layer_name_upper}_COARSE]);")
    
    hpp.close()

def generate_split_files(name, config, partition_name):
    if not os.path.exists(os.path.join(os.getcwd(), "generated_files", partition_name)):
        os.makedirs(os.path.join(os.getcwd(), "generated_files", partition_name))

    generate_split_hpp(name, config, partition_name)
    generate_split_cpp(name, config, partition_name)
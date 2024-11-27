import argparse
import cProfile
import logging
import pstats
import time

import yaml
from dotmap import DotMap

import wandb
from fpga_hart import _logger
from fpga_hart.layers.layer_parser import LayerParser
from fpga_hart.network.network_parser import NetworkParser
from fpga_hart.partitions.partition_parser import PartitionParser
from fpga_hart.platform.platform import Platform


def parse_args():
    """
    Argument parser function.
    """
    parser = argparse.ArgumentParser(description="fpga-hart toolflow parser")
    parser.add_argument(
        "model_name",
        choices=["x3d_m", "slowonly", "r2plus1d_18", "r2plus1d_34", "c3d", "unet"],
        type=str,
        help="name of the HAR model",
    )
    parser.add_argument(
        "device_name",
        choices=["zc706", "zcu104-106", "zcu102", "vc707", "vc709", "vus440"],
        type=str,
        help="name of the FPGA device",
    )
    parser.add_argument(
        "type",
        choices=["network", "partition", "layer"],
        type=str,
        help="type of processing to be performed",
    )
    parser.add_argument(
        "target",
        choices=["throughput", "latency"],
        type=str,
        help="target of the optimization",
    )
    parser.add_argument(
        "--singlethreaded",
        action="store_true",
        help="whether to use single thread solution or not",
    )
    parser.add_argument(
        "--se_block",
        action="store_true",
        help="whether to treat squeeze excitation as a block/layer or not",
    )
    parser.add_argument(
        "--plot_layers",
        action="store_true",
        help="whether to plot design points per layer or not",
    )
    parser.add_argument(
        "--gap_approx",
        action="store_true",
        help="whether to use historical data as approximation for GAP layers or not",
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="whether to run a (wandb) sweep of the design space or not",
    )
    parser.add_argument(
        "--enable_wandb",
        action="store_true",
        help="whether to enable wandb or not",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="whether to profile the whole programm or not",
    )
    return parser.parse_args()

def optimizer() -> None:
    args = parse_args()

    platform = Platform(args.device_name)

    project_name = f"fpga-hart-{args.model_name}-{args.type}-{args.target}"

    with open("fpga_hart/config/config_optimizer.yaml", "r") as yaml_file:
        config_dictionary = yaml.load(yaml_file, Loader=yaml.FullLoader)
        config_dictionary['device'] = platform.fpga_device
        config_dictionary['clock_frequency'] = platform.clock_freq
        config_dictionary['total_dsps'] = platform.dsp
        config_dictionary['total_brams'] = platform.bram
        config_dictionary['total_mem_bw'] = platform.mem_bw

    if args.enable_wandb:
        if args.sweep:
            wandb.init()
            config = wandb.config
            config.update({'device': platform.fpga_device})
            config.update({'clock_frequency': platform.clock_freq})
            config.update({'total_dsps': platform.dsp})
            config.update({'total_brams': platform.bram})
            config.update({'total_mem_bw': platform.mem_bw})
        else:
            wandb.init(config=config_dictionary, project=project_name)
            config = wandb.config
    else:
        config = DotMap(config_dictionary)

    if args.type == "network":
        network_parser = NetworkParser(
            model_name=args.model_name,
            se_block=args.se_block,
            batch_size=config.batch_size,
            num_reconfig_points=config.num_reconfig_points,
            allowed_reconfig_layers=config.allowed_reconfig_layers,
            min_partition_layers=config.min_partition_layers,
            max_partition_layers=config.max_partition_layers,
            gap_approx=args.gap_approx,
            platform=platform,
            config=config,
            enable_wandb=args.enable_wandb,
        )

        if args.target == "throughput":
            network_parser.parse()
        elif args.target == "latency":
            pass
    elif args.type == "partition":
        partition_parser = PartitionParser(
            model_name=args.model_name,
            se_block=args.se_block,
            gap_approx=args.gap_approx,
            singlethreaded=args.singlethreaded,
            per_layer_plot=args.plot_layers,
            platform=platform,
            config=config,
            enable_wandb=args.enable_wandb,
        )

        if args.target == "throughput":
            partition_parser.parse()
            # partition_parser.model_custom_partition(name="conv_relu_seq")
        elif args.target == "latency":
            # partition_parser.find_common_layers(groupping=3)
            partition_parser.latency_driven_design()
    elif args.type == "layer":
        layer_parser = LayerParser(
            model_name=args.model_name,
            se_block=args.se_block,
            singlethreaded=args.singlethreaded,
            per_layer_plot=args.plot_layers,
            platform=platform,
            config=config,
            enable_wandb=args.enable_wandb,
        )

        if args.target == "throughput":
            layer_parser.parse()
            # ["Conv", "Pool", "GlobalAveragePool", "Gemm", "Mul", "Add", "Relu", "Sigmoid", "Swish"]
            # layer_parser.model_custom_layer("Conv")
        elif args.target == "latency":
            pass
    else:
        raise ValueError("Invalid type of processing")

if __name__ == "__main__":
    start_time = time.time()

    args = parse_args()
    if args.profile:
        pr = cProfile.Profile()
        pr.enable()

    _logger.setLevel(level=logging.INFO)

    project_name = f"fpga-hart-{args.model_name}-{args.type}-{args.target}"

    if args.sweep:
        with open("fpga_hart/config/sweep_config.yaml", "r") as yaml_file:
            sweep_config = yaml.load(yaml_file, Loader=yaml.FullLoader)
            sweep_id = wandb.sweep(sweep_config, project=project_name)
        wandb.agent(sweep_id, function = optimizer)
    else:
        optimizer()


    end_time = time.time()
    _logger.info(
        msg=f"Total execution time: {time.strftime('%H:%M:%S', time.gmtime(end_time - start_time))}"
    )

    if args.profile:
        pr.disable()
        stats = pstats.Stats(pr).sort_stats('cumtime')
        stats.print_stats()

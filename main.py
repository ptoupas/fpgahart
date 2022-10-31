import argparse
import configparser
import cProfile
import logging
import os
import pstats
import time
from typing import Tuple

import seaborn as sns
import yaml

import wandb
from fpga_hart import _logger
from fpga_hart.layers.layer_parser import LayerParser
from fpga_hart.partitions.partition_parser import PartitionParser

sns.set(rc={"figure.figsize": (15, 8)})
sns.set_style("whitegrid")


def parse_args():
    """
    Argument parser function.
    """
    parser = argparse.ArgumentParser(description="fpga-hart toolflow parser")
    parser.add_argument(
        "model_name",
        choices=["x3d_m", "slowonly", "r2plus1d", "c3d"],
        type=str,
        help="name of the HAR model",
    )
    parser.add_argument(
        "type",
        choices=["partition", "layer"],
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
        "--nonalignedfactors",
        action="store_true",
        help="whether to allow the use of folding factors not perfectly divisible by channels/filters or not",
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


def get_fpga_specs() -> Tuple[str, int, int, int, float]:
    config = configparser.ConfigParser()
    config.read(os.path.join(os.getcwd(), "fpga_hart", "config", "config_fpga.ini"))

    word_length = int(config.get("FPGA Specifications", "word_length"))
    clock_freq = int(config.get("FPGA Specifications", "clock_freq"))
    bram = int(config.get("FPGA Specifications", "bram"))
    bram_Kbytes = int(config.get("FPGA Specifications", "bram_type")) / 8
    dsp = int(config.get("FPGA Specifications", "dsp"))
    mem_bw = float(config.get("FPGA Specifications", "mem_bw"))
    fpga_device = config.get("FPGA Specifications", "fpga_device")

    return fpga_device, clock_freq, dsp, bram, mem_bw

def optimizer() -> None:
    args = parse_args()

    project_name = f"fpga-hart-{args.model_name}-{args.type}-{args.target}"

    with open("fpga_hart/config/config_optimizer.yaml", "r") as yaml_file:
        config_dictionary = yaml.load(yaml_file, Loader=yaml.FullLoader)
        fpga_device, clock_freq, dsp, bram, mem_bw = get_fpga_specs()
        config_dictionary['device'] = fpga_device
        config_dictionary['clock_frequency'] = clock_freq
        config_dictionary['total_dsps'] = dsp
        config_dictionary['total_brams'] = bram
        config_dictionary['total_mem_bw'] = mem_bw


    config = None
    if args.enable_wandb:
        if args.sweep:
            wandb.init()
            config = wandb.config
            config.update({'device': fpga_device})
            config.update({'clock_frequency': clock_freq})
            config.update({'total_dsps': dsp})
            config.update({'total_brams': bram})
            config.update({'total_mem_bw': mem_bw})
        else:
            wandb.init(config=config_dictionary, project=project_name)
            config = wandb.config

    if args.type == "partition":
        partition_parser = PartitionParser(
            model_name=args.model_name,
            se_block=args.se_block,
            gap_approx=args.gap_approx,
            singlethreaded=args.singlethreaded,
            per_layer_plot=args.plot_layers,
            wandb_config=config,
        )

        if args.target == "throughput":
            partition_parser.parse()
            # partition_parser.model_custom_partition()
        elif args.target == "latency":
            # partition_parser.find_common_layers(groupping=3)
            partition_parser.latency_driven_design(
                plot_summaries=False,
                alignedfactors=config_dictionary['alignedfactors'] if args.enable_wandb else not args.nonalignedfactors,
                wandb_config=config,
            )
    elif args.type == "layer":
        layer_parser = LayerParser(
            model_name=args.model_name,
            se_block=args.se_block,
            singlethreaded=args.singlethreaded,
            per_layer_plot=args.plot_layers,
            wandb_config=config,
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

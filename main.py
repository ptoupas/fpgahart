import argparse
import logging

import seaborn as sns
import yaml
from boto import config

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
    parser.add_argument("model_name", help="name of the HAR model")
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
        help="whether to allow the use of coarse factors not perfectly divisible by channels/filters or not",
    )
    parser.add_argument(
        "--disable_wandb",
        action="store_false",
        help="whether to disable wandb or not",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    _logger.setLevel(level=logging.INFO)

    with open("fpga_hart/config/config_param.yaml", "r") as yaml_file:
        config_dictionary = yaml.load(yaml_file, Loader=yaml.FullLoader)

    project_name = f"fpga-hart-{args.model_name}-{args.type}-{args.target}"

    config = None
    if args.disable_wandb:
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
                run_name="latency_driven_modeling",
                plot_summaries=False,
                alignedfactors=not args.nonalignedfactors,
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
            # layer_parser.parse()
            layer_parser.model_custom_layer()
        elif args.target == "latency":
            pass

    else:
        raise ValueError("Invalid type of processing")

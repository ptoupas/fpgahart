import argparse
import logging

import seaborn as sns

from fpgaHART import _logger
from fpgaHART.layers.layer_parser import LayerParser
from fpgaHART.partitions.partition_parser import PartitionParser

sns.set(rc={"figure.figsize": (15, 8)})
sns.set_style("whitegrid")


def parse_args():
    """
        Argument parser function.
    """
    parser = argparse.ArgumentParser(description="fpgaHART toolflow parser")
    parser.add_argument("model_name", help="name of the HAR model")
    parser.add_argument(
        "type",
        choices=["partition", "layer"],
        type=str,
        help="type of processing to be performed",
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

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    _logger.setLevel(level=logging.INFO)

    if args.type == "partition":
        partition_parser = PartitionParser(
            model_name=args.model_name,
            se_block=args.se_block,
            gap_approx=args.gap_approx,
            singlethreaded=args.singlethreaded,
            per_layer_plot=args.plot_layers,
        )

        # partition_parser.parse()
        # partition_parser.model_custom_partition()
        # partition_parser.find_common_layers(groupping=3)
        partition_parser.group_conv_layers()
    elif args.type == "layer":
        layer_parser = LayerParser(
            model_name=args.model_name,
            se_block=args.se_block,
            singlethreaded=args.singlethreaded,
            per_layer_plot=args.plot_layers,
        )

        # layer_parser.parse()
        layer_parser.model_custom_layer()

    else:
        raise ValueError("Invalid type of processing")

import argparse

import seaborn as sns

from fpgaHART.partitions.partition_parser import PartitionParser

sns.set(rc={"figure.figsize": (15, 8)})
sns.set_style("whitegrid")


def parse_args():
    parser = argparse.ArgumentParser(description="fpgaHART toolflow parser")
    parser.add_argument("model_name", help="name of the HAR model")
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

    parser = PartitionParser(
        args.model_name,
        args.singlethreaded,
        args.plot_layers,
        args.se_block,
        args.gap_approx,
    )

    # parser.model_custom_partition()
    # parser.model_individual_layers()
    parser.parse()

import csv
import logging
import os
from dataclasses import dataclass, field
from multiprocessing import Pool

import mlflow
import numpy as np
from fpgaHART import _logger
from matplotlib import pyplot as plt

from ..network_representation.model_descriptor import ModelLayerDescriptor
from ..utils import utils
from .layer_design import layer_design_points


def multithreaded_modeling(operation, input, pool):
    results = pool.starmap(operation, input)
    return results


@dataclass
class LayerParser(ModelLayerDescriptor):
    singlethreaded: bool = False
    per_layer_plot: bool = False

    def __post_init__(self) -> None:
        ModelLayerDescriptor.__post_init__(self)  # Initialize the parent class
        # _logger.setLevel(level=logging.DEBUG)

        if not os.path.exists(os.path.join(os.getcwd(), "fpga_modeling_reports")):
            os.makedirs(os.path.join(os.getcwd(), "fpga_modeling_reports"))

        self.layer_model_file = os.path.join(
            os.getcwd(), "fpga_modeling_reports", self.model_name + "_layers.csv"
        )
        self.pareto_results = False
        if self.pareto_results:
            self.layer_model_file_par = os.path.join(
                os.getcwd(),
                "fpga_modeling_reports",
                self.model_name + "_layers_pareto.csv",
            )

    def model_layer(self, layer: str, layer_description: dict) -> None:
        _logger.info("Modeling {} layer...".format(layer))
        (
            throughput_gops,
            throughput_vols,
            latency,
            dsp_util,
            bram_util,
        ) = layer_design_points(
            layer, layer_description, self.layer_model_file, self.singlethreaded
        )

    def parse(self) -> None:

        with open(self.layer_model_file, mode="w") as layer_dp:
            csv_writer = csv.writer(
                layer_dp, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
            )
            csv_writer.writerow(
                [
                    "Layer",
                    "Latency(C)-No-Depth",
                    "Latency(C)",
                    "Latency(S)",
                    "GOP/s",
                    "GOPs",
                    "volumes/s",
                    "DSP(%)",
                    "BRAM(%)",
                    "RateIn",
                    "RateOut",
                    "Depth",
                    "Branch Depth",
                    "Muls",
                    "Adds",
                    "Mem(W)",
                    "Mem(KB)",
                    "DataSizeIn(MB)",
                    "DataSizeOut(MB)",
                    "MemBoundIn",
                    "MemBoundOut",
                    "config",
                ]
            )

        if self.pareto_results:
            with open(self.layer_model_file_par, mode="w") as layer_dp:
                csv_writer = csv.writer(
                    layer_dp, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
                )
                csv_writer.writerow(
                    [
                        "Layer",
                        "Latency(C)-No-Depth",
                        "Latency(C)",
                        "Latency(S)",
                        "GOP/s",
                        "GOPs",
                        "volumes/s",
                        "DSP(%)",
                        "BRAM(%)",
                        "RateIn",
                        "RateOut",
                        "Depth",
                        "Branch Depth",
                        "Muls",
                        "Adds",
                        "Mem(W)",
                        "Mem(KB)",
                        "DataSizeIn(MB)",
                        "DataSizeOut(MB)",
                        "MemBoundIn",
                        "MemBoundOut",
                        "config",
                    ]
                )

        for name, descriptor in self.layers.items():
            self.model_layer(name, descriptor)

        if self.pareto_results:
            utils.drop_duplicates_csv(self.layer_model_file)
            utils.get_paretto_csv(self.layer_model_file_par, self.layer_model_file)
            if self.per_layer_plot:
                utils.plot_layers_csv(self.layer_model_file_par, self.model_name)

    def model_custom_layer(self) -> None:
        with open(self.layer_model_file, mode="w") as layer_dp:
            csv_writer = csv.writer(
                layer_dp, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
            )
            csv_writer.writerow(
                [
                    "Layer",
                    "Latency(C)-No-Depth",
                    "Latency(C)",
                    "Latency(S)",
                    "GOP/s",
                    "GOPs",
                    "volumes/s",
                    "DSP(%)",
                    "BRAM(%)",
                    "RateIn",
                    "RateOut",
                    "Depth",
                    "Branch Depth",
                    "Muls",
                    "Adds",
                    "Mem(W)",
                    "Mem(KB)",
                    "DataSizeIn(MB)",
                    "DataSizeOut(MB)",
                    "MemBoundIn",
                    "MemBoundOut",
                    "config",
                ]
            )

        name = "custom_conv_layer"
        conv_descriptor = {
            "operation": "Conv",
            "shape_in": [[1, 24, 16, 32, 32]],
            "shape_out": [1, 12, 16, 32, 32],
            "node_in": ["575"],
            "node_out": "576",
            "branching": False,
            "kernel": [12, 24, 3, 3, 3],
            "bias": [12],
            "padding": [1, 1, 1],
            "stride": [1, 1, 1],
            "groups": 1,
            "dilation": [1, 1, 1],
        }
        self.model_layer(name, conv_descriptor)

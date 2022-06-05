import csv
import logging
import os
from dataclasses import dataclass, field
from multiprocessing import Pool

import mlflow
import numpy as np
from fpga_hart import _logger
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

        if not os.path.exists(
                os.path.join(os.getcwd(), "fpga_modeling_reports")):
            os.makedirs(os.path.join(os.getcwd(), "fpga_modeling_reports"))

        self.layer_model_file = os.path.join(os.getcwd(),
                                             "fpga_modeling_reports",
                                             self.model_name + "_layers.csv")
        self.pareto_results = False
        if self.pareto_results:
            self.layer_model_file_par = os.path.join(
                os.getcwd(),
                "fpga_modeling_reports",
                self.model_name + "_layers_pareto.csv",
            )

    def model_layer(self, layer: str, layer_description: dict) -> None:
        _logger.info("Modeling {} layer...".format(layer))
        layer_design_points(layer, layer_description, self.layer_model_file,
                            self.singlethreaded)

    def parse(self) -> None:

        with open(self.layer_model_file, mode="w") as layer_dp:
            csv_writer = csv.writer(layer_dp,
                                    delimiter=",",
                                    quotechar='"',
                                    quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow([
                "Layer",
                "Type",
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
            ])

        if self.pareto_results:
            with open(self.layer_model_file_par, mode="w") as layer_dp:
                csv_writer = csv.writer(layer_dp,
                                        delimiter=",",
                                        quotechar='"',
                                        quoting=csv.QUOTE_MINIMAL)
                csv_writer.writerow([
                    "Layer",
                    "Type",
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
                ])

        for name, descriptor in self.layers.items():
            self.model_layer(name, descriptor)

        if self.pareto_results:
            utils.drop_duplicates_csv(self.layer_model_file)
            utils.get_paretto_csv(self.layer_model_file_par,
                                  self.layer_model_file)
            if self.per_layer_plot:
                utils.plot_layers_csv(self.layer_model_file_par,
                                      self.model_name)

    def model_custom_layer(self) -> None:
        with open(self.layer_model_file, mode="w") as layer_dp:
            csv_writer = csv.writer(layer_dp,
                                    delimiter=",",
                                    quotechar='"',
                                    quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow([
                "Layer",
                "Type",
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
            ])

        name = "custom_conv_layer_dw"
        # conv_descriptor = {
        #     "operation": "Conv",
        #     "shape_in": [[1, 3, 8, 16, 16]],
        #     # "shape_out": [1, 6, 8, 16, 16],
        #     "shape_out": [1, 6, 8, 8, 8],
        #     "node_in": ["575"],
        #     "node_out": "576",
        #     "branching": False,
        #     "kernel": [6, 3, 3, 3, 3],
        #     "bias": [6],
        #     "padding": [1, 1, 1],
        #     "stride": [1, 2, 2],
        #     "groups": 1,
        #     "dilation": [1, 1, 1],
        # }
        conv_descriptor = {
            "operation": "Conv",
            "shape_in": [[1, 4, 4, 6, 6]],
            "shape_out": [1, 4, 4, 6, 6],
            "node_in": ["575"],
            "node_out": "576",
            "branching": False,
            "kernel": [4, 1, 3, 3, 3],
            "bias": [4],
            "padding": [1, 1, 1],
            "stride": [1, 1, 1],
            "groups": 4,
            "dilation": [1, 1, 1],
        }
        self.model_layer(name, conv_descriptor)

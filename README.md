# FPGA-HART

## Prerequisites
Create a python environment with python version >=3.7 and install the dependencies from `requirements.txt` file.
> pip install -r requirements.txt
## Library basic usage

> python main.py {cnn-model-onnx-file-name} {optimization-type} {optimization-target}

- `cnn-model-onnx-file-name`: Name of the CNN model to be optimized. The model should be provided in an .onnx file and placed under the folder `models/`. The the value of this parameters should be the name of the .onnx file without the suffix.
- `optimization-type`: This argument accepts only two values **partition and layer**. It dictates the type of optimization to be performed. Whether to optimize the model completely as a whole or a specific layer of the model.
- `optimization-target`: This argument accepts only two values **throughput and latency**. Dictates the target of the optimization process.

### Additional arguments
- `--gap_approx`: Whether to use historical data as approximation for GlobalAveragePooling (GAP) layers or not during the optimization process.
- `--nonalignedfactors`: Whether to allow the use of folding factors not perfectly divisible by channels/filters or not.
- `--enable_wandb`: Whether to enable wandb support or not (enabled by default).


> ./backend_pipeline.sh (only works for *throughput* optimization-target)

Automates the process of optimization data generation and creation of the backend code files for HLS. At the start of the scripts there are some variables that need to be defined.
- `MODEL_NAME`: Same as `cnn-model-onnx-file-name` argument
- `EXECUTION_TYPE`: Same as `optimization-type` argument
- `TARGET`: Same as `optimization-target` argument
- `CONFIG_FILE`: The output file path of the the optimization process (only for throughput optimization-target)
- `PARTITION_NAME`: The name of the final folder containing the partitions of the model
- `HLS_PARENT_DIR`: The path of the HLS folder in which the backend code will be generated

> ./layer_hls_project.sh (only works for *throughput* optimization-target)

The arguments are the same as in the `backend_pipeline.sh` case.

## Configuration files
The configuration files for specifying the FPGA device characteristics (`config_fpga.ini`), the wandb and optimizer parameters (`config_optimizer.yaml`), and the pytorch/onnx supported layers and parameters (`config_pytorch.ini`) are all under the following path `$fpga-hart/fpga_hart/config/`.
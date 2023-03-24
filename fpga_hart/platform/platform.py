import configparser
import os


class Platform:
    def __init__(self, device_name) -> None:
        self.get_fpga_specs(device_name)

    def get_fpga_specs(self, device_name):
        config = configparser.ConfigParser()
        config.read(os.path.join(os.getcwd(), "fpga_hart", "config", "config_fpga.ini"))

        available_fpga_devices = config.get("FPGA Device", "fpga_devices")
        self.fpga_device = device_name.upper()
        assert self.fpga_device in available_fpga_devices, f"{self.fpga_device} is not one of the supported FPGA devices"

        self.word_length = int(config.get(self.fpga_device, "word_length"))
        self.word_bytes = self.word_length / 8
        self.clock_freq = int(config.get(self.fpga_device, "clock_freq"))
        self.cycles_per_sec = self.clock_freq * 1e6
        self.bram = int(config.get(self.fpga_device, "bram"))
        self.bram_Kbytes = int(config.get(self.fpga_device, "bram_type")) / 8
        self.dsp = int(config.get(self.fpga_device, "dsp"))
        self.mem_bw = float(config.get(self.fpga_device, "mem_bw"))
        self.mem_bandwidth = self.mem_bw * 1e9
        self.mem_words_per_cycle = (
            self.mem_bandwidth / self.word_length
        ) / self.cycles_per_sec
        self.reconfiguration_time = float(
            config.get(self.fpga_device, "reconfiguration_time")
        )
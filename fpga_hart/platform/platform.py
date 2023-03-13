import configparser
import os


class Platform:
    def __init__(self) -> None:
        self.get_fpga_specs()

    def get_fpga_specs(self):
        config = configparser.ConfigParser()
        config.read(os.path.join(os.getcwd(), "fpga_hart", "config", "config_fpga.ini"))

        self.word_length = int(config.get("FPGA Specifications", "word_length"))
        self.word_bytes = self.word_length / 8
        self.clock_freq = int(config.get("FPGA Specifications", "clock_freq"))
        self.cycles_per_sec = self.clock_freq * 1e6
        self.bram = int(config.get("FPGA Specifications", "bram"))
        self.bram_Kbytes = int(config.get("FPGA Specifications", "bram_type")) / 8
        self.dsp = int(config.get("FPGA Specifications", "dsp"))
        self.mem_bw = float(config.get("FPGA Specifications", "mem_bw"))
        self.mem_bandwidth = self.mem_bw * 1e9
        self.mem_words_per_cycle = (
            self.mem_bandwidth / self.word_length
        ) / self.cycles_per_sec
        self.fpga_device = config.get("FPGA Specifications", "fpga_device")
        self.reconfiguration_time = float(
            config.get("FPGA Specifications", "reconfiguration_time")
        )
import configparser
import os
import math

class BaseLayer():
    def __init__(self, data_format='NDHWC'):
        assert data_format=='NDHWC' or data_format=='NCDHW', "Wrong data format. Accepted formats are 'NDHWC' or 'NCDHW'"
        
        self.get_config()
        self.word_bytes = self.word_length/8
        self.cycles_per_sec = self.clock_freq*1e6

    def get_config(self):
        config = configparser.ConfigParser()
        config.read(os.path.join(os.getcwd(), 'fpgaHART', 'config', 'config_fpga.ini'))

        self.word_length = config.get('FPGA Specifications', 'word_length')
        self.clock_freq = config.get('FPGA Specifications', 'clock_freq')
        self.bram = config.get('FPGA Specifications', 'bram')
        self.bram_bytes = config.get('FPGA Specifications', 'bram_type') / 8
        self.dsp = config.get('FPGA Specifications', 'dsp')
        self.mem_bw = config.get('FPGA Specifications', 'mem_bw')

    def get_dp_performance(self, rin, rout, muls, adds, mem):
        mem_kb = (mem * self.word_bytes) / 1e3
        mem_bram = math.ceil(mem_kb / self.bram_bytes)
        bram_util = (mem_bram / self.bram) * 100

        dsps_util = (muls/self.dsp)*100

        thr_in = self.cycles_per_sec * rin    # Words per second
        thr_out = self.cycles_per_sec * rout    # Words per second

        return thr_in, thr_out, dsps_util, bram_util
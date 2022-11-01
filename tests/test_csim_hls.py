import os
import subprocess
import unittest

from ddt import data, ddt


def check_csim_log(file_path):
    with open(file_path, 'r') as f:
        for line in f:
            if 'CSim done with 0 errors.' in line:
                return True
    return False

@ddt
class TestCsim(unittest.TestCase):
    @data(*[d for d in os.listdir(os.getcwd()) if os.path.isdir(d) and d != '__pycache__'])
    def test_csim_correctness(self, layer):

        # p = subprocess.Popen(["vitis_hls", "-f", "run_hls.tcl"], cwd=f"./{layer}").wait()

        break_flag = False
        csim_log = None
        for root, dirs, files in os.walk(os.path.join(os.getcwd(), layer)):
            for f in files:
                if 'csim.log' in f:
                    csim_log = os.path.join(root, f)
                    self.assertTrue(check_csim_log(csim_log))
                    break_flag = True
                    break
            if break_flag:
                break
        self.assertIsNotNone(csim_log)

# python -m unittest validate_hls.py -v
if __name__ == "__main__":
    unittest.main()

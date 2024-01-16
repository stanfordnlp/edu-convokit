import unittest
from edu_convokit import utils

class TestUtilsPreprocess(unittest.TestCase):
    def test_convert_time_hh_mm_ss_to_seconds_zero(self):
        time = "00:00:00"
        seconds = utils.convert_time_hh_mm_ss_to_seconds(time)
        self.assertEqual(seconds, 0)

    def test_convert_time_hh_mm_ss_to_seconds(self):
        time = "00:10:50"
        seconds = utils.convert_time_hh_mm_ss_to_seconds(time)
        self.assertEqual(seconds, 650)

if __name__ == '__main__':
    unittest.main()

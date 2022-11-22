import sys
import unittest
from typing import Tuple
import os
import pandas as pd
from src.mlengine.data_manager import file_reader


class FileReaderGetDfDetailsTestCase(unittest.TestCase):
    # https://docs.python.org/3/library/unittest.html#setupclass-and-teardownclass
    _test_df = None
    FILEPATH = r"tests"
    FILENAME = "cereal.csv"
    FILE = os.path.join(FILEPATH, FILENAME)

    @classmethod
    def setUpClass(cls):
        cls._test_df = pd.read_csv(FileReaderGetDfDetailsTestCase.FILE)

    @classmethod
    def tearDownClass(cls):
        del cls._test_df

    def test_get_df_details(self):
        self.assertTrue(isinstance(file_reader.get_df_details(FileReaderGetDfDetailsTestCase._test_df), Tuple))
        self.assertEqual(len(file_reader.get_df_details(FileReaderGetDfDetailsTestCase._test_df)), 5)


if __name__ == "__main__":
    unittest.main()

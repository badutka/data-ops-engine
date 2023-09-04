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
        cls._test_df = pd.read_csv(cls.FILE)

    @classmethod
    def tearDownClass(cls):
        del cls._test_df

    def test_get_df_details(self):
        result = file_reader.get_df_details(self._test_df)
        self.assertTrue(isinstance(result, Tuple))
        self.assertEqual(len(result), 5)


if __name__ == "__main__":
    unittest.main()

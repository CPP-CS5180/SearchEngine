import unittest
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Preprocessor import preprocess, preprocess_documents


class TestPreprocessor(unittest.TestCase):

    pass


if __name__ == '__main__':
    unittest.main()
import unittest
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Loader import load_documents, load_queries, load_qrels

DOCS_PATH    = 'dataset/documents.json'
QUERIES_PATH = 'dataset/queries.json'
QRELS_PATH   = 'dataset/qrels.json'


class TestLoader(unittest.TestCase):
    pass


if __name__ == '__main__':
    unittest.main()
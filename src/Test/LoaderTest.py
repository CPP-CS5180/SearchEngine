import unittest
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Loader import load_documents, load_queries, load_qrels

BASE_DIR     = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
DOCS_PATH    = os.path.join(BASE_DIR, 'dataset/documents.json')
QUERIES_PATH = os.path.join(BASE_DIR, 'dataset/queries.json')
QRELS_PATH   = os.path.join(BASE_DIR, 'dataset/qrels.json')


class TestLoader(unittest.TestCase):

    def test_load_documents_returns_dataframe(self):
        df = load_documents(DOCS_PATH)
        self.assertIsInstance(df, pd.DataFrame)

    def test_load_documents_has_correct_columns(self):
        df = load_documents(DOCS_PATH)
        self.assertIn('doc_id', df.columns)
        self.assertIn('text', df.columns)

    def test_load_documents_not_empty(self):
        df = load_documents(DOCS_PATH)
        self.assertGreater(len(df), 0)

    def test_load_queries_returns_dataframe(self):
        df = load_queries(QUERIES_PATH)
        self.assertIsInstance(df, pd.DataFrame)

    def test_load_queries_has_correct_columns(self):
        df = load_queries(QUERIES_PATH)
        self.assertIn('query_id', df.columns)
        self.assertIn('text', df.columns)

    def test_load_queries_not_empty(self):
        df = load_queries(QUERIES_PATH)
        self.assertGreater(len(df), 0)

    def test_load_qrels_returns_dataframe(self):
        df = load_qrels(QRELS_PATH)
        self.assertIsInstance(df, pd.DataFrame)

    def test_load_qrels_has_correct_columns(self):
        df = load_qrels(QRELS_PATH)
        self.assertIn('query_id', df.columns)
        self.assertIn('doc_id', df.columns)
        self.assertIn('relevance', df.columns)

    def test_load_qrels_not_empty(self):
        df = load_qrels(QRELS_PATH)
        self.assertGreater(len(df), 0)

if __name__ == '__main__':
    unittest.main()
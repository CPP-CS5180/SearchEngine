import unittest
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Preprocessor import preprocess, preprocess_documents

class TestPreprocessor(unittest.TestCase):

    def test_preprocess_returns_list(self):
        result = preprocess(['Hello World'])
        self.assertIsInstance(result, list)

    def test_preprocess_lowercases(self):
        result = preprocess(['Hello World'])
        self.assertEqual(result[0], result[0].lower())

    def test_preprocess_removes_punctuation(self):
        result = preprocess(['Hello, World!'])
        self.assertNotIn(',', result[0])
        self.assertNotIn('!', result[0])

    def test_preprocess_removes_stopwords(self):
        result = preprocess(['this is a test'])
        tokens = result[0].split()
        self.assertNotIn('this', tokens)
        self.assertNotIn('is', tokens)
        self.assertNotIn('a', tokens)

    def test_preprocess_multiple_texts(self):
        result = preprocess(['first text', 'second text'])
        self.assertEqual(len(result), 2)

    def test_preprocess_empty_list(self):
        result = preprocess([])
        self.assertEqual(result, [])

    def test_preprocess_documents_returns_dataframe(self):
        df = pd.DataFrame({'doc_id': ['1', '2'], 'text': ['buy stocks now', 'invest in bonds']})
        result = preprocess_documents(df)
        self.assertIsInstance(result, pd.DataFrame)

    def test_preprocess_documents_has_correct_columns(self):
        df = pd.DataFrame({'doc_id': ['1', '2'], 'text': ['buy stocks now', 'invest in bonds']})
        result = preprocess_documents(df)
        self.assertIn('doc_id', result.columns)
        self.assertIn('tokenized_text', result.columns)

    def test_preprocess_documents_preserves_doc_ids(self):
        df = pd.DataFrame({'doc_id': ['1', '2'], 'text': ['buy stocks now', 'invest in bonds']})
        result = preprocess_documents(df)
        self.assertListEqual(list(result['doc_id']), ['1', '2'])

    def test_preprocess_documents_row_count(self):
        df = pd.DataFrame({'doc_id': ['1', '2'], 'text': ['buy stocks now', 'invest in bonds']})
        result = preprocess_documents(df)
        self.assertEqual(len(result), 2)

if __name__ == '__main__':
    unittest.main()
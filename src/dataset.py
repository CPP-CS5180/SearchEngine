import os

import pandas as pd

_BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
_DEFAULT_PATHS = {
    'documents': os.path.join(_BASE_DIR, 'dataset/documents.json'),
    'queries':   os.path.join(_BASE_DIR, 'dataset/queries.json'),
    'qrels':     os.path.join(_BASE_DIR, 'dataset/qrels.json'),
}


class Dataset:
    def __init__(self, docs_path=None, queries_path=None, qrels_path=None):
        """
        Reads from the dataset paths provided and loads in
        """
        self.documents = _load_documents(docs_path or _DEFAULT_PATHS['documents'])
        self.queries   = _load_queries(queries_path or _DEFAULT_PATHS['queries'])
        self.qrels     = _load_qrels(qrels_path or _DEFAULT_PATHS['qrels'])
        self._doc_index = self.documents.set_index('doc_id')

    def get_document(self, doc_id: str):
        """Return the document row for a given doc_id, or None if not found."""
        key = str(doc_id)
        if key not in self._doc_index.index:
            return None
        return self._doc_index.loc[key]


def _load_documents(path):
    return pd.read_json(path, dtype={'doc_id': str})

def _load_queries(path):
    return pd.read_json(path, dtype={'query_id': str})

def _load_qrels(path):
    return pd.read_json(path, dtype={'query_id': str, 'doc_id': str})
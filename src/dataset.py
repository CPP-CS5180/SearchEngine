import os
from collections import defaultdict

import pandas
import pandas as pd

from pathlib import Path

_BASE_DIR = str(Path(__file__).resolve().parent.parent)
_DEFAULT_PATHS = {
    "documents": os.path.join(_BASE_DIR, "dataset/documents.json"),
    "queries": os.path.join(_BASE_DIR, "dataset/queries.json"),
    "qrels": os.path.join(_BASE_DIR, "dataset/qrels.json"),
}


class Dataset:
    def __init__(self, docs_path=None, queries_path=None, qrels_path=None):
        """
        Reads from the dataset paths provided and loads in
        """
        self._documents = (
            _load_documents(docs_path or _DEFAULT_PATHS["documents"])
            .set_index("doc_id")
            .sort_index()
        )

        self._queries = (
            _load_queries(queries_path or _DEFAULT_PATHS["queries"])
            .set_index(["query_id", "text"])
            .sort_index()
        )

        # create qrels dict
        _qrels = _load_qrels(qrels_path or _DEFAULT_PATHS["qrels"])
        self._relevance_dictionary = self._create_relevance_dictionary(_qrels)

    @staticmethod
    def _create_relevance_dictionary(qrel_df: pandas.DataFrame) -> dict[str, set[str]]:
        """Creates a relevance dictionary from the given relevance judgments DataFrame"""
        relevance_dict: defaultdict[str, set[str]] = defaultdict(set)
        query_id: str
        doc_id: str
        relevance: int
        for query_id, doc_id, relevance in zip(
            qrel_df["query_id"], qrel_df["doc_id"], qrel_df["relevance"]
        ):
            if relevance > 0:
                relevance_dict[query_id].add(doc_id)
        return dict(relevance_dict)

    def get_relevance_dictionary(self) -> dict[str, set[str]]:
        return self._relevance_dictionary

    def get_query_id(self, query_text: str):
        """Return the query_id for a given query text, or None if not found."""
        key = str(query_text)
        try:
            result = self._queries.xs(key, level="text")
        except KeyError:
            return None
        return result.index[0]

    def get_query_text(self, query_id: str):
        """Return the text for a given query id, or None if not found."""
        key = str(query_id)
        try:
            result = self._queries.xs(key, level="query_id")
        except KeyError:
            return None
        return result.index[0]

    def get_document(self, doc_id: str):
        """Return the document row for a given doc_id, or None if not found."""
        key = str(doc_id)
        if key not in self._documents.index:
            return None
        return self._documents.loc[key]

    def get_documents(self):
        return self._documents

    def num_documents(self):
        return len(self._documents)


def _load_documents(path):
    return pd.read_json(path, dtype={"doc_id": str, "text": str})


def _load_queries(path):
    return pd.read_json(path, dtype={"query_id": str, "text": str})


def _load_qrels(path):
    return pd.read_json(path, dtype={"query_id": str, "doc_id": str, "relevance": int})

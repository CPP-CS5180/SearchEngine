import os

import pandas as pd
import pytest

from dataset import Dataset, _load_documents, _load_queries, _load_qrels

from pathlib import Path

BASE_DIR = str(Path(__file__).resolve().parent.parent)
DOCS_PATH = os.path.join(BASE_DIR, "dataset/documents.json")
QUERIES_PATH = os.path.join(BASE_DIR, "dataset/queries.json")
QRELS_PATH = os.path.join(BASE_DIR, "dataset/qrels.json")


# --- standalone loader functions ---


def test_load_documents_returns_dataframe():
    assert isinstance(_load_documents(DOCS_PATH), pd.DataFrame)


def test_load_documents_has_correct_columns():
    df = _load_documents(DOCS_PATH)
    assert "doc_id" in df.columns
    assert "text" in df.columns


def test_load_documents_not_empty():
    assert len(_load_documents(DOCS_PATH)) > 0


def test_load_queries_returns_dataframe():
    assert isinstance(_load_queries(QUERIES_PATH), pd.DataFrame)


def test_load_queries_has_correct_columns():
    df = _load_queries(QUERIES_PATH)
    assert "query_id" in df.columns
    assert "text" in df.columns


def test_load_queries_not_empty():
    assert len(_load_queries(QUERIES_PATH)) > 0


def test_load_qrels_returns_dataframe():
    assert isinstance(_load_qrels(QRELS_PATH), pd.DataFrame)


def test_load_qrels_has_correct_columns():
    df = _load_qrels(QRELS_PATH)
    assert "query_id" in df.columns
    assert "doc_id" in df.columns
    assert "relevance" in df.columns


def test_load_qrels_not_empty():
    assert len(_load_qrels(QRELS_PATH)) > 0


# --- Dataset class ---


@pytest.fixture(scope="module")
def loader():
    return Dataset()


def test_dataset_documents_indexed_by_doc_id(loader):
    assert isinstance(loader._documents, pd.DataFrame)
    assert loader._documents.index.name == "doc_id"
    assert "text" in loader._documents.columns


def test_dataset_queries_multiindexed_by_query_id_and_text(loader):
    assert isinstance(loader._queries, pd.DataFrame)
    assert list(loader._queries.index.names) == ["query_id", "text"]


def test_dataset_relevance_dictionary_is_dict_of_sets(loader):
    rel = loader.get_relevance_dictionary()
    assert isinstance(rel, dict)
    assert len(rel) > 0
    sample_query_id, sample_docs = next(iter(rel.items()))
    assert isinstance(sample_query_id, str)
    assert isinstance(sample_docs, set)
    assert all(isinstance(d, str) for d in sample_docs)


def test_dataset_relevance_dictionary_excludes_zero_relevance(loader):
    qrels = _load_qrels(QRELS_PATH)
    rel = loader.get_relevance_dictionary()
    # any (query_id, doc_id) with relevance == 0 should not appear in the dict
    zero_rows = qrels[qrels["relevance"] == 0]
    for query_id, doc_id in zip(zero_rows["query_id"], zero_rows["doc_id"]):
        assert doc_id not in rel.get(query_id, set())


def test_get_document_returns_row(loader):
    first_id = loader._documents.index[0]
    doc = loader.get_document(first_id)
    assert doc is not None
    assert doc["text"]


def test_get_document_missing_returns_none(loader):
    assert loader.get_document("nonexistent_id_xyz") is None


def test_get_document_accepts_int_id(loader):
    first_id = loader._documents.index[0]
    doc = loader.get_document(int(first_id))
    assert doc is not None


def test_get_documents_returns_indexed_frame(loader):
    docs = loader.get_documents()
    assert isinstance(docs, pd.DataFrame)
    assert docs.index.name == "doc_id"
    assert len(docs) == loader.num_documents()


def test_num_documents_matches_underlying_frame(loader):
    assert loader.num_documents() == len(loader._documents)
    assert loader.num_documents() > 0


def test_get_query_text_roundtrips_with_get_query_id(loader):
    query_id, text = loader._queries.index[0]
    assert loader.get_query_text(query_id) == text
    assert loader.get_query_id(text) == query_id


def test_get_query_id_missing_returns_none(loader):
    assert loader.get_query_id("this query text does not exist xyz") is None


def test_get_query_text_missing_returns_none(loader):
    assert loader.get_query_text("nonexistent_query_id_xyz") is None

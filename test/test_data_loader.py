import os

import pandas as pd
import pytest

from dataset import Dataset, _load_documents, _load_queries, _load_qrels

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DOCS_PATH    = os.path.join(BASE_DIR, 'dataset/documents.json')
QUERIES_PATH = os.path.join(BASE_DIR, 'dataset/queries.json')
QRELS_PATH   = os.path.join(BASE_DIR, 'dataset/qrels.json')


# --- standalone loader functions ---

def test_load_documents_returns_dataframe():
    assert isinstance(_load_documents(DOCS_PATH), pd.DataFrame)

def test_load_documents_has_correct_columns():
    df = _load_documents(DOCS_PATH)
    assert 'doc_id' in df.columns
    assert 'text' in df.columns

def test_load_documents_not_empty():
    assert len(_load_documents(DOCS_PATH)) > 0

def test_load_queries_returns_dataframe():
    assert isinstance(_load_queries(QUERIES_PATH), pd.DataFrame)

def test_load_queries_has_correct_columns():
    df = _load_queries(QUERIES_PATH)
    assert 'query_id' in df.columns
    assert 'text' in df.columns

def test_load_queries_not_empty():
    assert len(_load_queries(QUERIES_PATH)) > 0

def test_load_qrels_returns_dataframe():
    assert isinstance(_load_qrels(QRELS_PATH), pd.DataFrame)

def test_load_qrels_has_correct_columns():
    df = _load_qrels(QRELS_PATH)
    assert 'query_id' in df.columns
    assert 'doc_id' in df.columns
    assert 'relevance' in df.columns

def test_load_qrels_not_empty():
    assert len(_load_qrels(QRELS_PATH)) > 0


# --- DatasetLoader class ---

@pytest.fixture(scope='module')
def loader():
    return Dataset()

def test_dataset_loader_has_documents(loader):
    assert isinstance(loader.documents, pd.DataFrame)
    assert 'doc_id' in loader.documents.columns

def test_dataset_loader_has_queries(loader):
    assert isinstance(loader.queries, pd.DataFrame)
    assert 'query_id' in loader.queries.columns

def test_dataset_loader_has_qrels(loader):
    assert isinstance(loader.qrels, pd.DataFrame)
    assert 'doc_id' in loader.qrels.columns

def test_get_document_returns_row(loader):
    first_id = loader.documents['doc_id'].iloc[0]
    doc = loader.get_document(first_id)
    assert doc is not None
    assert doc['text']

def test_get_document_missing_returns_none(loader):
    assert loader.get_document('nonexistent_id_xyz') is None

def test_get_document_accepts_int_id(loader):
    first_id = loader.documents['doc_id'].iloc[0]
    doc = loader.get_document(int(first_id))
    assert doc is not None
import os

import pandas as pd

from data_loader import load_documents, load_queries, load_qrels

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DOCS_PATH = os.path.join(BASE_DIR, 'dataset/documents.json')
QUERIES_PATH = os.path.join(BASE_DIR, 'dataset/queries.json')
QRELS_PATH = os.path.join(BASE_DIR, 'dataset/qrels.json')


def test_load_documents_returns_dataframe():
    df = load_documents(DOCS_PATH)
    assert isinstance(df, pd.DataFrame)


def test_load_documents_has_correct_columns():
    df = load_documents(DOCS_PATH)
    assert 'doc_id' in df.columns
    assert 'text' in df.columns


def test_load_documents_not_empty():
    df = load_documents(DOCS_PATH)
    assert len(df) > 0


def test_load_queries_returns_dataframe():
    df = load_queries(QUERIES_PATH)
    assert isinstance(df, pd.DataFrame)


def test_load_queries_has_correct_columns():
    df = load_queries(QUERIES_PATH)
    assert 'query_id' in df.columns
    assert 'text' in df.columns


def test_load_queries_not_empty():
    df = load_queries(QUERIES_PATH)
    assert len(df) > 0


def test_load_qrels_returns_dataframe():
    df = load_qrels(QRELS_PATH)
    assert isinstance(df, pd.DataFrame)


def test_load_qrels_has_correct_columns():
    df = load_qrels(QRELS_PATH)
    assert 'query_id' in df.columns
    assert 'doc_id' in df.columns
    assert 'relevance' in df.columns


def test_load_qrels_not_empty():
    df = load_qrels(QRELS_PATH)
    assert len(df) > 0

import pandas as pd

from Preprocessor import preprocess, preprocess_documents


def test_preprocess_returns_list():
    result = preprocess(['Hello World'])
    assert isinstance(result, list)


def test_preprocess_lowercases():
    result = preprocess(['Hello World'])
    assert result[0] == result[0].lower()


def test_preprocess_removes_punctuation():
    result = preprocess(['Hello, World!'])
    assert ',' not in result[0]
    assert '!' not in result[0]


def test_preprocess_removes_stopwords():
    result = preprocess(['this is a test'])
    tokens = result[0].split()
    assert 'this' not in tokens
    assert 'is' not in tokens
    assert 'a' not in tokens


def test_preprocess_multiple_texts():
    result = preprocess(['first text', 'second text'])
    assert len(result) == 2


def test_preprocess_empty_list():
    result = preprocess([])
    assert result == []


def test_preprocess_documents_returns_dataframe():
    df = pd.DataFrame({'doc_id': ['1', '2'], 'text': ['buy stocks now', 'invest in bonds']})
    result = preprocess_documents(df)
    assert isinstance(result, pd.DataFrame)


def test_preprocess_documents_has_correct_columns():
    df = pd.DataFrame({'doc_id': ['1', '2'], 'text': ['buy stocks now', 'invest in bonds']})
    result = preprocess_documents(df)
    assert 'doc_id' in result.columns
    assert 'tokenized_text' in result.columns


def test_preprocess_documents_preserves_doc_ids():
    df = pd.DataFrame({'doc_id': ['1', '2'], 'text': ['buy stocks now', 'invest in bonds']})
    result = preprocess_documents(df)
    assert list(result['doc_id']) == ['1', '2']


def test_preprocess_documents_row_count():
    df = pd.DataFrame({'doc_id': ['1', '2'], 'text': ['buy stocks now', 'invest in bonds']})
    result = preprocess_documents(df)
    assert len(result) == 2
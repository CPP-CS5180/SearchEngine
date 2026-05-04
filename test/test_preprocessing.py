import pandas as pd

from preprocessor import preprocess
from search_engine import SearchEngine


def test_preprocess_returns_list():
    result = preprocess(["Hello World"])
    assert isinstance(result, list)


def test_preprocess_lowercases():
    result = preprocess(["Hello World"])
    assert result[0] == result[0].lower()


def test_preprocess_removes_punctuation():
    result = preprocess(["Hello, World!"])
    assert "," not in result[0]
    assert "!" not in result[0]


def test_preprocess_removes_stopwords():
    result = preprocess(["this is a test"])
    tokens = result[0].split()
    assert "this" not in tokens
    assert "is" not in tokens
    assert "a" not in tokens


def test_preprocess_multiple_texts():
    result = preprocess(["first text", "second text"])
    assert len(result) == 2


def test_preprocess_empty_list():
    result = preprocess([])
    assert result == []


def test_tokenize_docs_returns_dict():
    df = pd.DataFrame(
        {"text": ["buy stocks now", "invest in bonds"]},
        index=pd.Index(["1", "2"], name="doc_id"),
    )
    result = SearchEngine.tokenize_docs(df)
    assert isinstance(result, dict)


def test_tokenize_docs_keys_are_doc_ids():
    df = pd.DataFrame(
        {"text": ["buy stocks now", "invest in bonds"]},
        index=pd.Index(["1", "2"], name="doc_id"),
    )
    result = SearchEngine.tokenize_docs(df)
    assert set(result.keys()) == {"1", "2"}


def test_tokenize_docs_values_are_lists():
    df = pd.DataFrame(
        {"text": ["buy stocks now", "invest in bonds"]},
        index=pd.Index(["1", "2"], name="doc_id"),
    )
    result = SearchEngine.tokenize_docs(df)
    for v in result.values():
        assert isinstance(v, list)


def test_tokenize_docs_size():
    df = pd.DataFrame(
        {"text": ["buy stocks now", "invest in bonds"]},
        index=pd.Index(["1", "2"], name="doc_id"),
    )
    result = SearchEngine.tokenize_docs(df)
    assert len(result) == 2


def test_tokenize_docs_empty_dataframe():
    df = pd.DataFrame(
        {"text": []},
        index=pd.Index([], name="doc_id", dtype=str),
    )
    result = SearchEngine.tokenize_docs(df)
    assert result == {}

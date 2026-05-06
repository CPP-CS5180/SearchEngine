import math
import os
from collections import Counter, defaultdict

import pandas as pd
from loguru import logger

from dataset import Dataset
from preprocessor import preprocess

DEFAULT_BM25_K1 = 1.8
DEFAULT_BM25_B = 0.75
DEFAULT_PER_PAGE = 5


class SearchEngine:
    def __init__(
        self,
        dataset_path: str,
        bm25_k1: float = DEFAULT_BM25_K1,
        bm25_b: float = DEFAULT_BM25_B,
    ):
        """
        Does not load the dataset or construct the inverted index yet, call init() to do that
        :param dataset_path:
        """
        self._dataset_path = dataset_path
        self._dataset: Dataset | None = None
        self._inverted_index: dict[str, list[tuple[str, int]]] | None = None
        self._query_cache: dict[str, list[str]] = {}  # query_id -> list[doc_id]
        self._doc_lengths: dict[str, int] = {}
        self._avg_doc_length: float = 0.0
        self._bm25_k1 = bm25_k1
        self._bm25_b = bm25_b

    def init(self):
        """
        Call this method to load the inverted index
        :return:
        """
        logger.info(f"Initializing search engine with path {self._dataset_path}.")

        logger.info("Loading dataset...")
        dataset = self._dataset = Dataset(
            docs_path=os.path.join(self._dataset_path, "documents.json"),
            queries_path=os.path.join(self._dataset_path, "queries.json"),
            qrels_path=os.path.join(self._dataset_path, "qrels.json"),
        )

        logger.info("Tokenizing documents...")
        tokenized_docs = self.tokenize_docs(
            dataset.get_documents()
        )

        self._doc_lengths = {
            doc_id: len(tokens) for doc_id, tokens in tokenized_docs.items()
        }
        self._avg_doc_length = (
            sum(self._doc_lengths.values()) / len(self._doc_lengths)
            if self._doc_lengths
            else 0.0
        )

        logger.info("Creating inverted index...")
        self._inverted_index = self._create_inverted_index(tokenized_docs)

    def get_dataset(self) -> Dataset:
        if self._dataset is None:
            raise ValueError("dataset not loaded yet, call init() first")
        return self._dataset

    def get_inverted_index(self) -> dict[str, list[tuple[str, int]]]:
        if self._inverted_index is None:
            raise ValueError("inverted index not created yet, call init() first")
        return self._inverted_index

    def search(
        self, query: str, page_index: int = 0, results_per_page: int = DEFAULT_PER_PAGE
    ) -> tuple[dict[str, str], int]:
        """
        :param query:
        :param page_index:
        :param results_per_page:
        :return: tuple (result -> document text, number of total results for the query)
        """
        if self._dataset is None:
            raise ValueError("dataset not loaded yet, call init() first")

        if self._inverted_index is None:
            raise ValueError("inverted index not created yet, call init() first")

        results = self._query_cache.setdefault(
            query, self._match_and_rank(SearchEngine._tokenize_text(query))
        )

        total_results: int = len(results)

        results = self._paginate(results, page_index, results_per_page)

        return {
            doc_id: self._dataset.get_document(doc_id)["text"] for doc_id in results
        }, total_results

    def set_bm25_params(self, *, bm25_k1: float = None, bm25_b: float = None):
        if bm25_k1 is not None:
            self._bm25_k1 = bm25_k1
        if bm25_b is not None:
            self._bm25_b = bm25_b
        if bm25_k1 is not None or bm25_b is not None:
            self._query_cache.clear()

    @staticmethod
    def _paginate(
        results: list[str], page_index: int, results_per_page: int
    ) -> list[str]:
        """
        :param results: list of doc_ids matching the query terms, ranked by relevance
        :param page_index: 0-indexed page number to return
        :param results_per_page: number of results to return per page
        :return: list of doc_ids for the requested page
        """
        if page_index < 0 or results_per_page <= 0:
            raise ValueError("page_index must be >= 0 and results_per_page must be > 0")
        start = page_index * results_per_page
        end = start + results_per_page
        return results[start:end]

    def _match_and_rank(self, tokenized_query: list[str]) -> list[str]:
        """
        Match and rank documents using Okapi BM25.

        :param tokenized_query: tokenized query terms
        :return: list of doc_ids ranked by BM25 score (most relevant first)
        """
        if self._inverted_index is None:
            raise ValueError("inverted index not created yet, call init() first")

        n_docs = len(self._doc_lengths)
        scores: defaultdict[str, float] = defaultdict(float)

        for term in set(tokenized_query):
            postings = self._inverted_index.get(term)
            if not postings:
                continue

            df = len(postings)
            idf = math.log((n_docs - df + 0.5) / (df + 0.5) + 1.0)

            self._fill_relevance_scores_from_postings(idf, postings, scores)

        return sorted(scores, key=lambda d: scores[d], reverse=True)

    def _fill_relevance_scores_from_postings(
        self,
        idf: float,
        postings: list[tuple[str, int]],
        scores: defaultdict[str, float],
    ):
        for doc_id, tf in postings:
            dl = self._doc_lengths[doc_id]
            norm = 1.0 - self._bm25_b + self._bm25_b * dl / self._avg_doc_length
            scores[doc_id] += (
                idf * (tf * (self._bm25_k1 + 1.0)) / (tf + self._bm25_k1 * norm)
            )

    @staticmethod
    def _create_inverted_index(
        tokenized_docs: dict[str, list[str]],
    ) -> dict[str, list[tuple[str, int]]]:
        """
        :param tokenized_docs: dict mapping doc_id -> list of tokenized terms
        :return: dict mapping term -> list of (doc_id, term_frequency) tuples
        e.g. {"hello": [("doc_1", 3), ("doc_2", 2)], "world": [("doc_1", 2)]}
        """
        inverted_index: dict[str, list[tuple[str, int]]] = defaultdict(list)

        for doc_id, tokens in tokenized_docs.items():
            for token, count in Counter(tokens).items():
                inverted_index[token].append((doc_id, count))

        return dict(inverted_index)

    @staticmethod
    def tokenize_docs(docs: pd.DataFrame) -> dict[str, list[str]]:
        """
        :param docs: a pandas dataframe of documents indexed by "doc_id" with a "text" column
        :return: mapping doc_id -> list of tokenized terms for the document in the corpus matching that doc_id
        """
        tokenized_docs: dict[str, list[str]] = dict()
        doc_id: str
        text: str
        for doc_id, text in zip(docs.index, docs["text"]):
            tokenized_docs[doc_id] = SearchEngine._tokenize_text(text)
        return tokenized_docs

    @staticmethod
    def _tokenize_text(text: str) -> list[str]:
        return preprocess([text])

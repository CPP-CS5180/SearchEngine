import pandas
from loguru import logger

from dataset import Dataset


class SearchEngine:

    def __init__(self, dataset_path: str):
        """
        Does not load the dataset or construct the inverted index yet, call init() to do that
        :param dataset_path:
        """
        self._dataset_path = dataset_path
        self._dataset_loader: Dataset | None = None
        # construct inverted index
        self._inverted_index: dict[str, list[tuple[str, int]]] = {}

    def init(self):
        """
        Call this method to load the inverted index
        :return:
        """
        logger.info(f"Initializing search engine with path {self._dataset_path}.")
        logger.info(f"Loading dataset...")
        self._dataset_loader = Dataset(self._dataset_path, self._dataset_path, self._dataset_path)
        logger.info(f"Creating inverted index...")
        self._inverted_index = self._get_inverted_index(self._tokenize_docs())

    def get_dataset(self) -> Dataset:
        if self._dataset_loader is None:
            raise ValueError("dataset not loaded yet, call init() first")
        return self._dataset_loader

    def get_inverted_index(self) -> dict[str, list[tuple[str, int]]]:
        if self._inverted_index is None:
            raise ValueError("inverted index not created yet, call init() first")
        return self._inverted_index

    def search(self, query: str, page: int = 0) -> dict[str, str]:
        if self._dataset_loader is None:
            raise ValueError("dataset not loaded yet, call init() first")

        if self._inverted_index is None:
            raise ValueError("inverted index not created yet, call init() first")

        # TODO: implement search and return {doc_id: text, ...}
        # TODO: implement rank and pagination as well
        return {
            "doc_001": "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.",
            "doc_042": "Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.",
            "doc_108": "Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur.",
            "doc_256": "Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.",
            "doc_512": "Sed ut perspiciatis unde omnis iste natus error sit voluptatem accusantium doloremque laudantium.",
        }

    def _tokenize_docs(self) -> pandas.DataFrame:
        """
        :return:
            a pandas dataframe containing the tokenized documents. col 1 should be "doc_id", col 2 should be "tokenized_text"
        """
        tokenized_df: pandas.DataFrame = pandas.DataFrame(columns=["doc_id", "tokenized_text"])
        # TODO: tokenize documents and fill dataframe
        return tokenized_df

    def _get_inverted_index(self, tokenized_df: pandas.DataFrame) -> dict[str, list[tuple[str, int]]]:
        """
        :param tokenized_df: a pandas dataframe containing the tokenized documents. col 1 should be "doc_id", col 2 should be "tokenized_text"
        :return: dict mapping term -> list of (doc_id, term_frequency) tuples
        e.g. {"hello": [("doc_1", 3), ("doc_2", 2)], "world": [("doc_1", 2)]}
        """
        # construct inverted index and return it
        inverted_index = {}
        return inverted_index

import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Will be downloaded when the file is executed. (only one time download)
nltk.download("punkt_tab")
nltk.download("stopwords")
nltk.download("wordnet")

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


def preprocess(raw_texts: list[str]) -> list[str]:
    tokenized_texts = []
    for text in raw_texts:
        # lowercasing
        text = text.lower()

        # stripping punctuations/numbers
        text = re.sub(r"[^a-z\s]", "", text)

        # tokenization
        tokens = word_tokenize(text)

        # stopword removal
        tokens = [t for t in tokens if t not in stop_words and len(t) > 1]

        # lemmatization
        tokenized_texts = [lemmatizer.lemmatize(t) for t in tokens]

    return tokenized_texts

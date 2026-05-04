import re
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Will be downloaded when the file is executed. (only one time download)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

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
        tokens = [lemmatizer.lemmatize(t) for t in tokens]

        tokenized_texts.append(" ".join(tokens))
    return tokenized_texts
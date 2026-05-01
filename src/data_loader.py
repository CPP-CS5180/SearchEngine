import pandas as pd

def load_documents(path):
    return pd.read_json(path)

def load_queries(path):
    return pd.read_json(path)

def load_qrels(path):
    return pd.read_json(path)
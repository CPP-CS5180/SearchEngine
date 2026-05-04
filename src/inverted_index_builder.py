def inverted_index_maker(tokenized_dict: dict) -> dict:
    inverted_index = {}

    for doc_id, tokens in tokenized_dict.items():
        counts = [(token, tokens.count(token)) for token in set(tokens)]
        for token, count in counts:
            if token in inverted_index:
                inverted_index[token].append((doc_id, count))
            else:
                inverted_index[token] = [(doc_id, count)]

    return inverted_index

# SearchEngine — UML Class Diagram

```mermaid
classDiagram
    direction LR

    class SearchEngine {
        -str _dataset_path
        -Dataset _dataset
        -dict _inverted_index
        -dict _query_cache
        -dict _tokenized_docs
        -dict _doc_lengths
        -float _avg_doc_length
        -float _bm25_k1
        -float _bm25_b
        +__init__(dataset_path, bm25_k1, bm25_b)
        +init() void
        +get_dataset() Dataset
        +get_inverted_index() dict
        +search(query, page_index, results_per_page) tuple
        +set_bm25_params(bm25_k1, bm25_b) void
        -_match_relevant_docs(tokenized_query) set
        -_rank(tokenized_query, results) list
        -_relevance_scores_from_posting(...) void
        +tokenize_docs(docs)$ dict
        -_tokenize_text(text)$ list
        -_create_inverted_index(tokenized_docs)$ dict
        -_paginate(results, page_index, results_per_page)$ list
    }

    class Dataset {
        -DataFrame _documents
        -DataFrame _queries
        -dict _relevance_dictionary
        +__init__(docs_path, queries_path, qrels_path)
        +get_relevance_dictionary() dict
        +get_query_id(query_text) str
        +get_query_text(query_id) str
        +get_document(doc_id) Series
        +get_documents() DataFrame
        +num_documents() int
        -_create_relevance_dictionary(qrel_df)$ dict
    }

    class preprocessor {
        <<module>>
        +stop_words : set
        +lemmatizer : WordNetLemmatizer
        +preprocess(raw_texts) list
    }

    class ArgNamespace {
        +int port
        +str host
        +bool debug
        +str dataset_path
    }

    class parse_args_module {
        <<module>>
        +parse_args() ArgNamespace
    }

    class Main {
        <<module __main__>>
        +MAX_VISIBLE_PAGES : int
        +_build_pagination(current, total, max_visible) list
        +register_routes(app) void
        +setup_logger(debug) void
        +create_app(args) Flask
        +main() void
    }

    class FlaskApp {
        <<external>>
        +route("/") index()
        +route("/doc/<doc_id>") doc_view()
        +route("/api/v1/search") search()
    }

    Namespace <|-- ArgNamespace : extends
    SearchEngine *-- Dataset : composes
    SearchEngine ..> preprocessor : uses
    Main ..> SearchEngine : instantiates
    Main ..> parse_args_module : uses
    Main ..> ArgNamespace : uses
    Main ..> FlaskApp : creates
    parse_args_module ..> ArgNamespace : returns
    FlaskApp ..> SearchEngine : queries
```

## Notes

- `SearchEngine` composes a `Dataset` (built in `init()`) and delegates tokenization to the `preprocessor` module.
- `ArgNamespace` extends `argparse.Namespace` to provide typed CLI args; `parse_args()` returns it.
- `__main__` is the Flask entry point — it wires CLI args, builds the app, instantiates `SearchEngine`, and registers the `/`, `/doc/<doc_id>`, and `/api/v1/search` routes.
- `benchmark.py` is intentionally omitted.
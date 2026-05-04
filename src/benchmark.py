"""Evaluate the search engine against the dataset's relevance judgments."""
import argparse
import sys
import time

from loguru import logger

from search_engine import SearchEngine


def average_precision(retrieved: list[str], relevant: set[str]) -> float:
    if not relevant:
        return 0.0
    hits = 0
    score = 0.0
    for i, d in enumerate(retrieved, start=1):
        if d in relevant:
            hits += 1
            score += hits / i
    return score / len(relevant)


def run_benchmark(dataset_path: str) -> tuple[dict[str, tuple[str, float]], float]:
    """Returns (per-query (text, AP) keyed by query_id, MAP)."""
    engine = SearchEngine(dataset_path=dataset_path)
    engine.init()
    dataset = engine.get_dataset()
    qrels = dataset.get_relevance_dictionary()
    full_depth = dataset.num_documents()

    aps: dict[str, tuple[str, float]] = {}

    logger.info(f"Evaluating {len(qrels)} queries over the full corpus...")
    start = time.perf_counter()
    for query_id, relevant in qrels.items():
        query_text = dataset.get_query_text(query_id)
        if query_text is None or not relevant:
            continue
        page, _ = engine.search(query_text, page_index=0, results_per_page=full_depth)
        retrieved = list(page.keys())
        aps[query_id] = (query_text, average_precision(retrieved, relevant))

    elapsed = time.perf_counter() - start
    n = len(aps)
    logger.info(
        f"Evaluated {n} queries in {elapsed:.2f}s "
        f"({elapsed / n * 1000:.1f}ms/query)" if n else f"Evaluated 0 queries in {elapsed:.2f}s"
    )
    mean_ap = sum(ap for _, ap in aps.values()) / n if n else 0.0
    return aps, mean_ap


def main():
    parser = argparse.ArgumentParser(description="Benchmark the search engine using qrels")
    parser.add_argument("--dataset_path", type=str, default="../dataset")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    logger.remove()
    logger.add(sys.stderr, level="DEBUG" if args.debug else "INFO")

    aps, mean_ap = run_benchmark(args.dataset_path)

    print(f"\nPer-query AP ({len(aps)} queries):")
    print("-" * 60)
    for query_id, (query_text, ap) in sorted(aps.items(), key=lambda kv: kv[1][1], reverse=True):
        print(f"  {query_id:>8}  AP={ap:.4f}  {query_text}")
    print("-" * 60)
    print(f"  MAP: {mean_ap:.4f}")


if __name__ == "__main__":
    main()
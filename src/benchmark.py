"""Evaluate the search engine against the dataset's relevance judgments."""

import argparse
import sys
import time

from loguru import logger

from search_engine import SearchEngine

DEFAULT_K1_GRID = [0.5, 0.9, 1.2, 1.5, 1.8, 2.0, 2.5, 3.0]
DEFAULT_B_GRID = [0.0, 0.25, 0.5, 0.75, 1.0]


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


def evaluate(engine: SearchEngine) -> tuple[dict[str, tuple[str, float]], float]:
    """Returns (per-query (text, AP) keyed by query_id, MAP)."""
    dataset = engine.get_dataset()
    qrels = dataset.get_relevance_dictionary()
    full_depth = dataset.num_documents()

    aps: dict[str, tuple[str, float]] = {}

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
        f"Evaluated {n} queries in {elapsed:.2f}s ({elapsed / n * 1000:.1f}ms/query)"
        if n
        else f"Evaluated 0 queries in {elapsed:.2f}s"
    )
    mean_ap = sum(ap for _, ap in aps.values()) / n if n else 0.0
    return aps, mean_ap


def run_sweep(
    dataset_path: str, k1_values: list[float], b_values: list[float]
) -> dict[tuple[float, float], tuple[dict[str, tuple[str, float]], float]]:
    """Build the engine once and evaluate across a grid of (k1, b) values."""
    engine = SearchEngine(dataset_path=dataset_path)
    engine.init()

    results: dict[tuple[float, float], tuple[dict[str, tuple[str, float]], float]] = {}
    total = len(k1_values) * len(b_values)
    i = 0
    for k1 in k1_values:
        for b in b_values:
            i += 1
            engine.set_bm25_params(bm25_k1=k1, bm25_b=b)
            logger.info(f"[{i}/{total}] Evaluating bm25_k1={k1}, bm25_b={b}...")
            results[(k1, b)] = evaluate(engine)
    return results


def _parse_floats(s: str) -> list[float]:
    return [float(x) for x in s.split(",") if x.strip()]


def print_map_matrix(
    sweep: dict[tuple[float, float], tuple[dict[str, tuple[str, float]], float]],
    k1_values: list[float],
    b_values: list[float],
) -> None:
    print("\nMAP by (bm25_k1, bm25_b):")
    header = "  k1 \\ b   " + "".join(f"{b:>8.2f}" for b in b_values)
    print(header)
    print("-" * len(header))
    for k1 in k1_values:
        row = f"  {k1:>6.2f}   " + "".join(
            f"{sweep[(k1, b)][1]:>8.4f}" for b in b_values
        )
        print(row)


def print_per_query_ap(
    aps: dict[str, tuple[str, float]], mean_ap: float, title: str
) -> None:
    print(f"\n{title} ({len(aps)} queries):")
    print("-" * 60)
    for query_id, (query_text, ap) in sorted(
        aps.items(), key=lambda kv: kv[1][1], reverse=True
    ):
        print(f"  {query_id:>8}  AP={ap:.4f}  {query_text}")
    print("-" * 60)
    print(f"  MAP: {mean_ap:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark the search engine using qrels"
    )
    parser.add_argument("--dataset_path", type=str, default="../dataset")
    parser.add_argument(
        "--bm25_k1",
        type=_parse_floats,
        default=DEFAULT_K1_GRID,
        help="Comma-separated bm25_k1 values to sweep (e.g. '1.2,1.5,2.0')",
    )
    parser.add_argument(
        "--bm25_b",
        type=_parse_floats,
        default=DEFAULT_B_GRID,
        help="Comma-separated bm25_b values to sweep (e.g. '0.5,0.75,1.0')",
    )
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    logger.remove()
    logger.add(sys.stderr, level="DEBUG" if args.debug else "INFO")

    sweep = run_sweep(args.dataset_path, args.bm25_k1, args.bm25_b)

    print_map_matrix(sweep, args.bm25_k1, args.bm25_b)

    best_params, (best_aps, best_map) = max(sweep.items(), key=lambda kv: kv[1][1])
    best_k1, best_b = best_params
    print(f"\nBest: bm25_k1={best_k1}, bm25_b={best_b} -> MAP={best_map:.4f}")

    print_per_query_ap(
        best_aps,
        best_map,
        f"Per-query AP for best params (bm25_k1={best_k1}, bm25_b={best_b})",
    )


if __name__ == "__main__":
    main()

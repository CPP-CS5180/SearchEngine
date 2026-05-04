import sys

from flask import Flask, render_template, request, jsonify, abort
from loguru import logger

import parse_args
from parse_args import ArgNamespace
from search_engine import SearchEngine

MAX_DESCRIPTION_CHARS = 450

def register_routes(app: Flask):
    search_engine = SearchEngine(dataset_path=app.config["DATASET_PATH"])
    search_engine.init()

    @app.route("/")
    def index():
        q = request.args.get("q", "")
        page_num = int(request.args.get("page_num", 1))
        results_per_page = int(request.args.get("per_page", 25))
        if q:
            results, total_results = search_engine.search(
                q, page_index=page_num - 1, results_per_page=results_per_page
            )
        else:
            results, total_results = {}, 0
        results = {
            doc_id: (text if len(text) <= MAX_DESCRIPTION_CHARS
                     else text[:MAX_DESCRIPTION_CHARS].rstrip() + "...")
            for doc_id, text in results.items()
        }
        total_pages = max(1, (total_results + results_per_page - 1) // results_per_page)
        return render_template(
            "index.html.j2",
            q=q,
            results=results,
            page_num=page_num,
            per_page=results_per_page,
            total_pages=total_pages,
            total_results=total_results,
        )

    @app.route("/doc/<doc_id>")
    def doc_view(doc_id: str):
        document = search_engine.get_dataset().get_document(doc_id)
        if document is None:
            abort(404)
        return render_template("doc.html.j2", doc_id=doc_id, text=document["text"])

    @app.route("/api/v1/search", methods=["GET"])
    def search():
        query = request.args.get("q", "")
        page_num = int(request.args.get("page_num", 1))
        results_per_page = int(request.args.get("per_page", 25))
        logger.debug(f"search query: {query!r}")
        return jsonify(
            search_engine.search(
                query, page_index=page_num - 1, results_per_page=results_per_page
            )[0]
        )


def setup_logger(debug: bool):
    logger.remove()
    level = "DEBUG" if debug else "INFO"
    logger.add(sys.stderr, level=level)


def create_app(args: ArgNamespace) -> Flask:
    app = Flask(__name__, template_folder="../templates", static_folder="../static")
    app.config["DATASET_PATH"] = args.dataset_path
    register_routes(app)
    return app


def main():
    args = parse_args.parse_args()
    setup_logger(args.debug)
    app = create_app(args)
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()

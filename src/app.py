import sys

from flask import Flask, render_template, request, jsonify, abort

import data_loader
import parse_args
from loguru import logger

app = Flask(__name__, template_folder="../templates")


def _search_results(query: str) -> dict[str, str]:
    # TODO: implement search and return {doc_id: text, ...}
    # TODO: implement rank and pagination as well
    return {
        "doc_001": "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.",
        "doc_042": "Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.",
        "doc_108": "Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur.",
        "doc_256": "Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.",
        "doc_512": "Sed ut perspiciatis unde omnis iste natus error sit voluptatem accusantium doloremque laudantium.",
    }


@app.route('/')
def index():
    q = request.args.get('q', '')
    results = _search_results(q) if q else {}
    return render_template('index.html', q=q, results=results)


@app.route('/doc/<doc_id>')
def doc_view(doc_id: str):
    text = data_loader.get_document(app.config['DATASET_PATH'], doc_id)
    if text is None:
        abort(404)
    return render_template('doc.html', doc_id=doc_id, text=text)


@app.route('/api/search', methods=['GET'])
def search():
    query = request.args.get('q', '')
    logger.debug(f"search query: {query!r}")
    return jsonify(_search_results(query))

def setup_logger(debug: bool):
    logger.remove()
    level = "DEBUG" if debug else "INFO"
    logger.add(sys.stderr, level=level)

if __name__ == '__main__':
    args = parse_args.parse_args()
    setup_logger(args.debug)

    app.config['DATASET_PATH'] = args.dataset_path

    app.run(host=args.host, port=args.port, debug=args.debug)

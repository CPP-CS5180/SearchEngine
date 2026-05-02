import argparse


class ArgNamespace(argparse.Namespace):
    port: int
    host: str
    debug: bool
    dataset_path: str

def parse_args() -> ArgNamespace:
    parser = argparse.ArgumentParser(description="Search Engine")
    parser.add_argument(
        "--debug",
        "-d",
        action="store_true",
        help="enable debug mode",
        default=False,
    )
    parser.add_argument(
        "--port",
        "-p",
        type=int,
        default=8888,
        help="port number",
    )
    parser.add_argument(
        "--host",
        "-H",
        type=str,
        default="localhost",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="../dataset",
    )
    args = parser.parse_args(namespace=ArgNamespace())
    return args

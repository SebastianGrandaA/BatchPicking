from argparse import ArgumentParser
from logging import (
    DEBUG,
    INFO,
    WARNING,
    Formatter,
    StreamHandler,
    basicConfig,
    getLogger,
)
from os import scandir
from typing import Any

from app import run_describe, run_experiment, run_optimize

DEFAULT_TIMEOUT = 25 * 60


def initialize() -> Any:
    """Initialize the application and return the parameters."""
    parser = ArgumentParser(description="BatchPicking")
    parser.add_argument(
        "-u",
        "--use_case",
        type=str,
        help="Use case",
        required=True,
        choices=["optimize", "experiment", "describe"],
    )
    parser.add_argument(
        "-m",
        "--method",
        type=str,
        help="Method to use",
        required=True,
        choices=["sequential", "joint"],
    )
    parser.add_argument(
        "-n", "--instance_name", type=str, help="Instance name", required=False
    )
    parser.add_argument(
        "-ns", "--instance_names", type=str, help="Instance names", required=False
    )
    parser.add_argument(
        "-t",
        "--timeout",
        type=int,
        help="Timeout",
        required=False,
        default=DEFAULT_TIMEOUT,
    )
    parser.add_argument(
        "-l",
        "--log_level",
        type=str,
        help="Log level",
        required=False,
        default="DEBUG",
        choices=["DEBUG", "INFO"],
    )
    args = parser.parse_args()
    log_level = DEBUG if args.log_level.upper() == "DEBUG" else INFO

    getLogger("matplotlib").setLevel(WARNING)
    basicConfig(level=log_level)
    stream = StreamHandler()
    formatter = Formatter("[%(asctime)s] %(levelname)s : %(message)s", datefmt="%H:%M")
    stream.setFormatter(formatter)
    getLogger().handlers[0].setFormatter(formatter)

    return args


def dispatch(args: Any) -> None:
    """Dispatch the use case to the corresponding function."""
    dir_list = lambda name: [f.path for f in scandir(name) if f.is_dir()]

    if args.use_case == "optimize":
        run_optimize(args.method, args.instance_name, args.timeout)

    elif args.use_case == "experiment":
        if args.instance_names == "all":
            instances = []

            for warehouse in dir_list("data"):
                if warehouse == "-":
                    continue

                for subfolder in dir_list(warehouse):
                    name = subfolder.split("/")[1:]
                    instances.append("/".join(name))

        else:
            instances = args.instance_names.split(",")

        run_experiment(args.method, instances, args.timeout)

    elif args.use_case == "describe":
        run_describe()

    else:
        raise ValueError(f"Invalid use case: {args.use_case}")


if __name__ == "__main__":
    """Entry point for the application."""
    args = initialize()
    dispatch(args)

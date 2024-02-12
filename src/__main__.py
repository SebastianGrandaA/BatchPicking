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
    """
    Entry point for the application.
    It supports three use cases: optimize, experiment, and describe.

    The optimize use case is used to solve a single instance. For example:
        python src -u optimize -m joint -n examples/toy_instance -t 1500

    The experiment use case is used to solve multiple instances and compare the results. For example:
        python src -u experiment -m joint -ns all -t 1800 -l INFO
        python src -u experiment -m joint -ns examples/toy_instance,warehouse_A/data_2023-05-22,warehouse_B/data_2023-05-22,warehouse_C/2023-09-08_15-00-00_RACK-4,warehouse_D/data_2023-01-30_00 -t 1800 -l INFO
        python src -u experiment -m joint -ns warehouse_A/data_2023-05-23,warehouse_A/data_2023-05-24,warehouse_A/data_2023-05-25,warehouse_A/data_2023-05-26,warehouse_A/data_2023-05-27 -t 1800 -l INFO

    The describe use case is used to analyze the results of the optimization process. For example:
        python src -u describe -m joint
    """
    args = initialize()
    dispatch(args)

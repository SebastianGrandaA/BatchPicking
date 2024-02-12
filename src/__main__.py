from argparse import ArgumentParser
from logging import DEBUG, INFO, WARNING, basicConfig, getLogger
from os import scandir

from app import run_experiment, run_optimize

DEFAULT_TIMEOUT = 25 * 60


def initialize(log_level: int) -> None:
    getLogger("matplotlib").setLevel(WARNING)
    basicConfig(level=log_level)


if __name__ == "__main__":
    """
    Entry point for the application.
    It supports two use cases: optimize and experiment.

    The optimize use case is used to solve a single instance. For example:
        python src -u optimize -m joint -n examples/toy_instance -t 1500

    The experiment use case is used to solve multiple instances and compare the results. For example:
        python src -u experiment -m joint -ns examples/toy_instance,warehouse_A/data_2023-05-22,warehouse_B/data_2023-05-22,warehouse_C/2023-09-08_15-00-00_RACK-4,warehouse_D/data_2023-01-30_00 -t 3600 -l INFO
    """
    parser = ArgumentParser(description="BatchPicking")
    parser.add_argument(
        "-u",
        "--use_case",
        type=str,
        help="Use case",
        required=True,
        choices=["optimize", "experiment"],
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

    initialize(log_level)
    dir_list = lambda name: [f.path for f in scandir(name) if f.is_dir()]

    if args.use_case == "optimize":
        run_optimize(args.method, args.instance_name, args.timeout)

    elif args.use_case == "experiment":
        if args.instance_names == "all":
            for warehouse in dir_list("data"):
                for subfolder in dir_list(warehouse):
                    run_experiment(args.method, subfolder, args.timeout)

        else:
            run_experiment(args.method, args.instance_names.split(","), args.timeout)
    else:
        raise ValueError(f"Invalid use case: {args.use_case}")

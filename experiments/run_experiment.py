import sys
import os
import argparse
import matplotlib
matplotlib.use('Agg')
import asyncio
from enum import Enum

sys.path.append(os.path.abspath(os.path.join('..')))
import lib

RANDOM_STATE = 0
VALID_SOURCES = ['ToolE', 'Berkeley', 'ToolLens', 'ReverseChain']

def enum_by_name(enum_cls: Enum) -> callable:
    """
    Create a function that converts a string to an Enum member by its .name (case-insensitive).

    Args:
        enum_cls (Enum): The Enum class to create the converter for.

    Returns:
        callable: A function that takes a string and returns the corresponding Enum member.
    """

    lookup = {e.name.lower(): e for e in enum_cls}
    def convert(s: str):
        key = s.strip().lower()
        if key in lookup:
            return lookup[key]
        choices = ", ".join([e.name for e in enum_cls])
        raise argparse.ArgumentTypeError(
            f"Invalid {enum_cls.__name__}: {s}. Choose from: {choices}"
        )
    return convert

def positive_int(s: str) -> int:
    try:
        v = int(s)
    except ValueError:
        raise argparse.ArgumentTypeError("must be an integer")
    if v <= 0:
        raise argparse.ArgumentTypeError("must be > 0")
    return v

def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run an experiment with the specified settings."
    )
    parser.add_argument(
        "--experiment_type",
        "-e",
        type=enum_by_name(lib.experiment.ExperimentSetupType),
        required=True,
        help=f"Experiment setup. Options: {', '.join([e.name for e in lib.experiment.ExperimentSetupType])}",
    )
    parser.add_argument(
        "--source",
        "-s",
        type=str,
        choices=VALID_SOURCES,
        required=True,
        help=f"Dataset/source. Options: {', '.join(VALID_SOURCES)}",
    )
    parser.add_argument(
        "--client_type",
        "-c",
        type=enum_by_name(lib.agent.ClientType),
        required=True,
        help=f"Client backend. Options: {', '.join([e.name for e in lib.agent.ClientType])}",
    )
    parser.add_argument(
        "--model_name",
        "-m",
        type=str,
        required=True,
        help="Model name/ID to use (e.g., gpt-4.1, gpt-4o-mini, gpt-4.1-nano, llama-3.1-8b-instant, llama-3-70b, etc.)",
    )
    parser.add_argument(
        "--max_num_samples",
        "-n",
        type=positive_int,
        help="Optional cap on number of samples (> 0).",
    )
    parser.add_argument(
        "--concurrency_limit",
        "-cl",
        type=positive_int,
        default=32,
        help="Optional cap on number of concurrent tasks (> 0).",
    )
    return parser.parse_args(argv)

if __name__ == '__main__':
    args = parse_args(sys.argv[1:])

    experiment_type: lib.experiment.ExperimentSetupType = args.experiment_type
    client_type: lib.agent.ClientType = args.client_type
    model_name: str = args.model_name
    source: str = args.source
    max_num_samples: int = args.max_num_samples if args.max_num_samples else None
    concurrency_limit: int = args.concurrency_limit

    if client_type is None:
        raise ValueError(f"Client type must be specified.")
    if model_name is None:
        raise ValueError(f"Model name must be specified.")
    if source is None:
        raise ValueError(f"Source must be specified.")
    if experiment_type is None:
        raise ValueError(f"Experiment type must be specified.")

    (
        react_agent, 
        df_test_single, 
        df_test_multi, 
        df_tool_descriptions,
        df_tool_embeddings,
        prompt_single, 
        prompt_multi, 
        experiment_name, 
        plot_title, 
        description
    ) = lib.experiment.load_experiment_setup(
        setupType=experiment_type,
        client_type=client_type,
        model_name=model_name,
        source=source,
        num_samples=max_num_samples,
        random_state=RANDOM_STATE,
    )

    # concurrency_limit = 64 if client_type.value == lib.agent.ClientType.OPENAI.value else 8
    asyncio.run(
        lib.experiment.test_all_tasks(
            react_agent,
            df_test_single=df_test_single,
            df_test_multi=df_test_multi,
            prompt_single=prompt_single,
            prompt_multi=prompt_multi,
            description=description,
            experiment_name=experiment_name,
            data_source=source if source is not None else "Main",
            plot_title=plot_title,
            concurrency_limit=concurrency_limit,
        )
    )

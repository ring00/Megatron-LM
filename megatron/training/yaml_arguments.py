# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""Megatron arguments."""

import json
import os

from types import SimpleNamespace
import yaml, re, os

# Taken from https://stackoverflow.com/questions/65414773/parse-environment-variable-from-yaml-with-pyyaml
# Allows for yaml to use environment variables
env_pattern = re.compile(r".*?\${(.*?)}.*?")


def env_constructor(loader, node):
    value = loader.construct_scalar(node)
    for group in env_pattern.findall(value):
        assert os.environ.get(group) is not None, f"environment variable {group} in yaml not found"
        value = value.replace(f"${{{group}}}", os.environ.get(group))
    return value


yaml.add_implicit_resolver("!pathex", env_pattern)
yaml.add_constructor("!pathex", env_constructor)


def merge_yaml_and_cli_args(yaml_args, cli_args, ignore_unknown_args=False):
    args = SimpleNamespace()

    for key, cli_value in vars(cli_args).items():
        yaml_value = getattr(yaml_args, key, None)
        expected_type = type(cli_value)

        if yaml_value is not None:
            assert isinstance(
                yaml_value, expected_type
            ), f"Invalid type for '{key}' in YAML. Expected '{expected_type.__name__}', got '{type(yaml_value).__name__}'."
            setattr(args, key, yaml_value)
        else:
            setattr(args, key, cli_value)

    if not ignore_unknown_args:
        for key in vars(yaml_args):
            assert hasattr(cli_args, key), f"Unknown argument '{key}' in YAML config."

    return args


def load_yaml(yaml_path, cli_args, ignore_unknown_args=False):
    """Load and merge YAML configuration with command line arguments.

    This function reads a YAML configuration file and merges it with command line arguments,
    giving precedence to YAML configurations.

    Args:
        yaml_path (str): Path to the YAML configuration file.
        cli_args (Namespace): Command line arguments parsed into a namespace object.
        ignore_unknown_args (bool, optional): Whether to ignore unknown arguments during merge.
            Defaults to False.
    """

    print(
        "WARNING: Using experimental YAML argument feature, command line arguments will be overwritten."
    )
    with open(yaml_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        # Convert to nested namespace
        yaml_args = json.loads(json.dumps(config), object_hook=lambda item: SimpleNamespace(**item))
        return merge_yaml_and_cli_args(yaml_args, cli_args, ignore_unknown_args)

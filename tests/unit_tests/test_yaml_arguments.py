# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""Unit tests for yaml_arguments module."""

import json
import os
import tempfile
from types import SimpleNamespace

import pytest


class TestYAMLArguments:
    """Test YAML argument loading and merging functionality."""

    def test_merge_yaml_and_cli_args_yaml_overrides(self):
        """Test that YAML values override CLI values."""
        from megatron.training.yaml_arguments import merge_yaml_and_cli_args

        # Create CLI args
        cli_args = SimpleNamespace(
            learning_rate=0.001,
            batch_size=32,
            num_epochs=10
        )

        # Create YAML args with some overrides
        yaml_args = SimpleNamespace(
            learning_rate=0.01,  # Override
            batch_size=64,       # Override
        )

        result = merge_yaml_and_cli_args(yaml_args, cli_args, ignore_unknown_args=True)

        # Check that YAML values override CLI values
        assert result.learning_rate == 0.01
        assert result.batch_size == 64
        # Check that CLI value is used when YAML doesn't have it
        assert result.num_epochs == 10

    def test_merge_yaml_and_cli_args_type_validation(self):
        """Test that type validation works correctly."""
        from megatron.training.yaml_arguments import merge_yaml_and_cli_args

        cli_args = SimpleNamespace(learning_rate=0.001)
        yaml_args = SimpleNamespace(learning_rate="invalid_string")

        with pytest.raises(AssertionError, match="Invalid type"):
            merge_yaml_and_cli_args(yaml_args, cli_args)

    def test_merge_yaml_and_cli_args_unknown_arg_detection(self):
        """Test that unknown arguments are detected."""
        from megatron.training.yaml_arguments import merge_yaml_and_cli_args

        cli_args = SimpleNamespace(learning_rate=0.001)
        yaml_args = SimpleNamespace(
            learning_rate=0.01,
            unknown_arg="value"
        )

        # Should raise error when ignore_unknown_args=False
        with pytest.raises(AssertionError, match="Unknown argument"):
            merge_yaml_and_cli_args(yaml_args, cli_args, ignore_unknown_args=False)

        # Should not raise error when ignore_unknown_args=True
        result = merge_yaml_and_cli_args(yaml_args, cli_args, ignore_unknown_args=True)
        assert result.learning_rate == 0.01

    def test_load_yaml_basic(self):
        """Test basic YAML file loading."""
        from megatron.training.yaml_arguments import load_yaml

        # Create a temporary YAML file
        yaml_content = """
learning_rate: 0.01
batch_size: 64
num_epochs: 100
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            yaml_path = f.name

        try:
            cli_args = SimpleNamespace(
                learning_rate=0.001,
                batch_size=32,
                num_epochs=10
            )

            result = load_yaml(yaml_path, cli_args, ignore_unknown_args=True)

            assert result.learning_rate == 0.01
            assert result.batch_size == 64
            assert result.num_epochs == 100
        finally:
            os.unlink(yaml_path)

    def test_env_variable_substitution(self):
        """Test environment variable substitution in YAML."""
        from megatron.training.yaml_arguments import load_yaml

        # Set a test environment variable
        os.environ['TEST_VAR'] = '/test/path'

        yaml_content = """
data_path: ${TEST_VAR}/data
checkpoint_path: ${TEST_VAR}/checkpoints
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            yaml_path = f.name

        try:
            cli_args = SimpleNamespace(
                data_path='/default/data',
                checkpoint_path='/default/checkpoints'
            )

            result = load_yaml(yaml_path, cli_args, ignore_unknown_args=True)

            assert result.data_path == '/test/path/data'
            assert result.checkpoint_path == '/test/path/checkpoints'
        finally:
            os.unlink(yaml_path)
            del os.environ['TEST_VAR']

    def test_nested_namespace_conversion(self):
        """Test that nested dictionaries are converted to nested namespaces."""
        from megatron.training.yaml_arguments import load_yaml

        yaml_content = """
optimizer:
  type: adam
  lr: 0.001
model:
  layers: 12
  hidden_size: 768
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            yaml_path = f.name

        try:
            # Create CLI args that would match this structure
            optimizer_ns = SimpleNamespace(type='sgd', lr=0.01)
            model_ns = SimpleNamespace(layers=6, hidden_size=512)
            cli_args = SimpleNamespace(optimizer=optimizer_ns, model=model_ns)

            result = load_yaml(yaml_path, cli_args, ignore_unknown_args=True)

            # Check that nested structures are properly converted
            assert hasattr(result.optimizer, 'type')
            assert result.optimizer.type == 'adam'
            assert result.optimizer.lr == 0.001
            assert result.model.layers == 12
            assert result.model.hidden_size == 768
        finally:
            os.unlink(yaml_path)

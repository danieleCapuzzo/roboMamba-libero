"""
Frozen dataclass configs with a generated argparse overlay.
Configs must be comparable and serializable (`dataclasses.asdict`) so `--resume`
can assert the saved run's config matches the one just passed on the CLI.
"""

import argparse
import dataclasses
from pathlib import Path


def add_dataclass_args(parser: argparse.ArgumentParser, cls: type) -> None:
    """Adds one CLI flag per dataclass field, defaulting to the field's default.

    Args:
        parser: Target argparse parser.
        cls: A dataclass type (not instance).
    """
    for field in dataclasses.fields(cls):
        flag = "--" + field.name.replace("_", "-")
        if field.type is bool or field.default is True or field.default is False:
            parser.add_argument(flag, action=argparse.BooleanOptionalAction, default=field.default)
        elif field.default is dataclasses.MISSING:
            parser.add_argument(flag, required=True)
        else:
            # Path/int/float/str fields all accept a plain string-typed default;
            # dataclass __post_init__-free coercion happens in cfg_from_args below
            parser.add_argument(flag, default=field.default)


def cfg_from_args(cls: type, args: argparse.Namespace):
    """Builds a dataclass instance from parsed args, coercing Path fields.

    Args:
        cls: A dataclass type.
        args: Namespace produced by a parser built with `add_dataclass_args`.

    Returns:
        An instance of `cls`.
    """
    kwargs = {}
    for field in dataclasses.fields(cls):
        value = getattr(args, field.name)
        if value is not None and field.type is Path:
            value = Path(value)
        kwargs[field.name] = value
    return cls(**kwargs)

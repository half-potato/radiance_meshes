from argparse import ArgumentParser, Namespace
from pathlib import Path
import json

class Args:
    def __init__(self):
        self._data = {}

    def __setattr__(self, key, value):
        if key == "_data":
            super().__setattr__(key, value)
        else:
            self._data[key] = value

    def __getattr__(self, key):
        if key in self._data:
            return self._data[key]
        raise AttributeError(f"'Args' object has no attribute '{key}'")

    def as_dict(self):
        """Return the stored arguments as a dictionary."""
        return self._data

    def get_parser(self):
        """Generate an ArgumentParser with stored values as defaults."""
        parser = ArgumentParser(description="Argument parser for the script")
        for key, value in self._data.items():
            if isinstance(value, bool):
                # Special handling for boolean flags
                parser.add_argument(
                    f"--{key}",
                    action="store_true" if not value else "store_false",
                    default=value
                )
            elif isinstance(value, list):
                # --- NEW: Handling for list arguments ---
                parser.add_argument(
                    f"--{key}",
                    nargs='+', # Accepts one or more arguments
                    default=value
                )
            else:
                # Handling for other types (int, float, str, etc.)
                parser.add_argument(f"--{key}", type=type(value), default=value)
        return parser

    @classmethod
    def from_namespace(cls, namespace: Namespace):
        """Convert a parsed Namespace back into an Args object."""
        obj = cls()
        obj._data = vars(namespace)
        return obj

    @classmethod
    def load_from_json(cls, json_path: str):
        """Load arguments from a JSON file."""
        path = Path(json_path)
        if not path.exists():
            raise FileNotFoundError(f"JSON file not found: {json_path}")

        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        obj = cls()
        obj._data = data
        return obj

    def __add__(self, other):
        """Merges two Args objects, updating the first with the second."""
        if not isinstance(other, Args):
            raise TypeError("Can only add Args objects together.")

        new_args = Args()
        # Other's values overwrite self's values in case of conflict
        new_args._data = {**self._data, **other._data}
        return new_args

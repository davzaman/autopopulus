from argparse import ArgumentParser, Namespace, Action
from enum import Enum
import re
import sys
import yaml
from os.path import isfile
from collections import ChainMap
from typing import Dict, List, Optional
from json import loads
from logging import warn


def load_cli_args(args_options_path: str = "options.yml"):
    """
    Modify command line args if desired, or load from YAML file.
    """
    if isfile(args_options_path):  # if file exists
        with open(args_options_path, "r") as f:
            res = yaml.safe_load(f)

        # sys.argv = [sys.argv[0]]
        for k, v in res.items():
            sys.argv += [f"--{k}", str(v)]


def parse_guild_args(obj, base_prefix=""):
    """This can be used for debugging.
    The results of loading the guild yml file is a list of dictionaries.
    Each item in the list is an object in the file represented as a dict.
        -config: common-flags will become {"config": "common-flags", ...}
        -flags: k: v, k: v, ... will be {"flags": {k: v, k: v}}
        This all goes into the same dict object.
    For multiple config groupings we'll have different objects.
    """
    # Grabs the config objects from the yaml file and then merges them.
    # the if: grab only config objects.
    # the chainmap will merge all the flag dictionaries from each group.
    #   if it encounters the same name later, it keeps the first one
    flags = dict(
        ChainMap(*[flag_group["flags"] for flag_group in obj if "config" in flag_group])
    )

    for k, v in flags.items():
        # ingore the $includes, because bash will think it's a var
        if k != "$include":
            # if the yaml has a list, just pick the first one for testing purposes
            if isinstance(v, list):
                v = v[0]
            # print('{}="{}"'.format(k.replace("-", "_"), v))
    return {k: v[0] if isinstance(v, list) else v for k, v in flags.items()}


def str2bool(str_value: str, default: bool = False):
    """Convert argparse boolean to true boolean.
    Ref: https://stackoverflow.com/a/43357954/1888794"""
    if isinstance(str_value, bool):
        return str_value
    if str_value.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif str_value.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        warn(
            f"Boolean value expected. Passed {str_value}, which is of type {type(str_value)} and wasn't recognized. Using default value {default}.",
        )
        return default


def StringToEnum(enum: Enum):
    class ConvertToEnum(Action):
        """Takes a string that corresponds to the name of an attribute of the given enum."""

        def __call__(
            self,
            parser: ArgumentParser,
            namespace: Namespace,
            value: str,
            option_string: Optional[str] = None,
        ):
            setattr(namespace, self.dest, enum[value.upper()])

    return ConvertToEnum


def string_json_to_python(obj_string: str) -> Dict:
    return loads(obj_string.replace("'", '"'))


def YAMLStringListToList(convert: type = str, choices: Optional[List[str]] = None):
    class ConvertToList(Action):
        """Takes a comma separated list (no spaces) from command line and parses into list of some type (Default str)."""

        def __call__(
            self,
            parser: ArgumentParser,
            namespace: Namespace,
            values: str,
            option_string: Optional[str] = None,
        ):

            if convert == dict:  # Internal dict will be a string
                values = string_json_to_python(values)
                setattr(namespace, self.dest, values)
                return

            # strip any {<space>, ', " [, ]}" and then split by comma
            values = re.sub(r"[ '\"\[\]]", "", values).split(",")
            if choices:
                values = [convert(x) for x in values if x in choices]
            else:
                values = [convert(x) for x in values]
            setattr(namespace, self.dest, values)

    return ConvertToList


def YAMLStringDictToDict(
    choices: Optional[List[str]] = None,
):
    class ConvertToDict(Action):
        """Takes a comma separated list (no spaces) from command line and parses into list of some type (Default str)."""

        def __call__(
            self,
            parser: ArgumentParser,
            namespace: Namespace,
            dict_string: str,
            option_string: Optional[str] = None,
        ):
            if choices:
                dict_obj = {
                    key: value
                    for key, value in string_json_to_python(dict_string).items()
                    if key in choices
                }
            else:
                dict_obj = string_json_to_python(dict_string)
            setattr(namespace, self.dest, dict_obj)

    return ConvertToDict

"""
One-off run passing the yaml stuff to guild directly (or else it won't see it.)
"""
import subprocess
import yaml
from os.path import isfile

FORCE_FLAGS: bool = False
DEBUG: bool = True


def cli_str(obj) -> str:
    # format for split() also same thign for CLI. the str repr of lists/etc has spaces which will create problems.
    str_rep = str(obj).replace(" ", "")
    if isinstance(obj, list):
        # list wrapped in quotes, or else guild thinks it's a grid
        str_rep = str_rep.replace('"', "'")
        return f'"{str_rep}"'
    return str_rep


if __name__ == "__main__":
    guild_path: str = "guild.yml"
    with open(guild_path, "r") as f:
        guild_setup = yaml.safe_load(f)

    guild_expected_flags = list(guild_setup[0]["flags"].keys())

    args_options_path: str = "options.yml"
    if isfile(args_options_path):  # if file exists
        with open(args_options_path, "r") as f:
            options = yaml.safe_load(f)

    # get values from options.yml that are the high level flags in guild
    command_args = {
        name: options[name] for name in guild_expected_flags if name in options
    }

    command = "guild run main".split()
    command += [f"{name}={cli_str(val)}" for name, val in command_args.items()]
    if FORCE_FLAGS:
        command.append("--force-flags")
    if DEBUG:
        command.append("--debug-sourcecode=.")
    subprocess.run(command)

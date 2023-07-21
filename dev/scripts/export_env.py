from argparse import ArgumentParser
import subprocess
import yaml

try:
    import pip_chill
except ModuleNotFoundError:
    print("module 'pip-chill' is not installed, but is required for this script.")

if __name__ == "__main__":
    p = ArgumentParser()
    p.add_argument("--env-file", type=str, default="env.yml")
    args = p.parse_known_args()[0]

    # Export manually installed conda/mamba packages to env.yml
    mamba_pkgs = subprocess.run(
        ["mamba", "env", "export", "--from-history"], stdout=subprocess.PIPE
    ).stdout.decode("utf-8")
    yml = yaml.safe_load(mamba_pkgs)
    # manually installed pip packages, requires pip-chill
    pip_pkgs = (
        subprocess.run(["pip-chill", "--no-version"], stdout=subprocess.PIPE)
        .stdout.decode("utf-8")
        .split()
    )
    # packages installed by pip but not mamba (unique)
    pip_but_not_mamba = list(set(pip_pkgs) - set(yml["dependencies"]))
    # add pip-installed packages to yaml
    if "pip" not in yml["dependencies"]:  # need to install pip manually in the list
        yml["dependencies"].append("pip")
    # https://stackoverflow.com/a/35245610/1888794
    yml["dependencies"].append({"pip": pip_but_not_mamba})

    # Dump results
    with open(args.env_file, "w") as f:
        yaml.dump(yml, f)
    print("Done.")

    """
    From here:
    1. mamba deactivate
    2. mamba env remove -n <env>
    3. mamba clean -a  # clean cache
    3. mamba env create --file <env_file>
    """

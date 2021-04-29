import yaml
import sys

from imputer import main
from utils.utils import parse_yaml_args

if __name__ == "__main__":
    """Pass in either just the name of the method, or nothing."""
    if len(sys.argv) <= 3:
        with open("guild.yml", "r") as f:
            res = parse_yaml_args(yaml.safe_load(f))

        if len(sys.argv) == 3:
            method = sys.argv[-1]
        else:
            method = "ap_new"

        sys.argv = [sys.argv[0]]
        res["method"] = method
        # res["num-gpus"] = 1
        res["runtest"] = False
        res["runtune"] = False
        res["tune_n_samples"] = 1

        res["max-epochs"] = 3
        res["num-bootstraps"] = 3

        # For enforcing not fully observed even if its in the guild file
        enforce_full = False
        if enforce_full:
            del res["fully-observed"]
            del res["percent-missing"]
            del res["missingness-mechanism"]

        # res["percent-missing"] = 0.66
        res["missingness-mechanism"] = "MNAR"

        for k, v in res.items():
            sys.argv += [f"--{k.replace('_','-')}", str(v)]

    main()

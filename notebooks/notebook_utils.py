import sys, os
import torch

sys.path.insert(1, os.path.join(sys.path[0], "path/to/autopopulus"))

from autopopulus.data.utils import get_subgroup, load_features_and_labels, ampute
from autopopulus.utils.utils import parse_yaml_args
from autopopulus.imputer import init_args
import yaml


def set_args_for_notebook():
    with open("guild.yml", "r") as f:
        res = parse_yaml_args(yaml.safe_load(f))

    sys.argv = [sys.argv[0], "ap_new"]
    d = {f"--{k.replace('_','-')}": str(v) for k, v in res.items()}
    if d["--fully-observed"] == "False":
        del d["--fully-observed"]
    sys.argv += [item for k in d for item in (k, d[k])]
    args = init_args()
    args.val = True
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Pick device
    use_cuda = torch.cuda.is_available()
    # device = torch.device("cuda:3" if use_cuda else "cpu")
    device = torch.device("cuda" if use_cuda else "cpu")

    return (args, device)


def cleanup():
    # cleanup
    os.system("rm -r F.O.")
    os.system("rm -r full")
    os.system("rm -r serialized_models")

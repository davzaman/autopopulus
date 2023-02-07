import sys
import subprocess
import re

output = subprocess.run(
    [sys.executable, "autopopulus/impute.py"], stdout=subprocess.PIPE
).stdout.decode("utf-8")
aim_hash = re.search(r"(?:Aim Logger Hash: )(\w+)\b", str(output)).groups()[-1]
subprocess.run([sys.executable, "autopopulus/predict.py", "--aim-hash", aim_hash])

# TODO: no tuning working yet, no grid, and no evaluate

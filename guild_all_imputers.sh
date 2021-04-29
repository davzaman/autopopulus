# !/bin/bash

# https://my.guild.ai/t/running-cases-in-parallel/341
guild run fo --stage-trials
guild run all-imputers --stage-trials
for _ in `seq 10`; do guild run queue --background -y; done
guild view
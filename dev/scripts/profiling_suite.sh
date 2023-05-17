python -m cProfile -o profiling-results/program.prof autopopulus/utils/profile_autoencoder.py

fil-profile run autopopulus/utils/profile_autoencoder.py

# run pytorch-lightning profiling, must manually open text files
# advanced is the same as cProfile above and the one above allows me to see nested calls
# the pytorch profiler cuts off names if they're too long
python autopopulus/utils/profile_autoencoder.py --profilers=[simple,pytorch]

# visualize pytorch profiling with notebooks/visualize_profiling.ipynb

# visualize profile of whole program with snakeviz
snakeviz profiling-results/program.prof
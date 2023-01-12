python -m cProfile -o profiling-results/program.prof autopopulus/utils/profile_autoencoder.py

fil-profile run autopopulus/utils/profile_autoencoder.py

# run pytorch-lightning profiling, must manually open text files
python autopopulus/utils/profile_autoencoder.py --profilers=[simple,advanced,pytorch]

# visualize pytorch profiling with notebooks/visualize_profiling.ipynb

# visualize profile of whole program with snakeviz
snakeviz profiling-results/program.prof
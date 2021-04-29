#!/usr/bin/env python
from distutils.core import setup

setup(
    name="Autopopulus",
    version="1.0",
    description="Imputation suite that uses autoencoders for imputation.",
    author="Davina Zamanzadeh",
    author_email="davzaman@gmail.com",
    packages=["autopopulus"],
    requires=[
        "pytorch>=1.8",
        "tensorboardX",
        "rich",  # nice stack traces + printing
        "xgboost",
        "miceforest",
        "pytorch-lightning",
        "pandas",
        "numpy",
        "scikit-learn",
        "orange3",
        "tqdm",
        "pyarrow",  # if using feather preproc
        "keras",  # if using keras for dnn
    ],
    # ray for tuning, guild for tracking experiments, or notebook figures
    extras_require={"tune_track": ["ray", "guildai", "plotly", "nbformat"]},
)

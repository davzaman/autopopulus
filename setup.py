#!/usr/bin/env python
from distutils.core import setup

setup(
    name="Autopopulus",
    version="2.0",
    description="Imputation suite that uses autoencoders for imputation.",
    author="Davina Zamanzadeh",
    author_email="davzaman@gmail.com",
    packages=["autopopulus"],
    install_requires=[
        "python>=3.5,<=3.9",  # abstract class / other python abilities, but 3.10 is too new for some packages
        "pytorch-lightning",
        "pandas>=1.3.0",  # for multicol explode
        "torch>=1.8",  # nan_to_num
        "xgboost",
        "rich",  # nice stack traces + printing
        "miceforest",
        "numpy",
        "scikit-learn",
        "orange3",
        "tqdm",
        "pyarrow",  # if using feather preproc
        "keras",  # if using keras for dnn
        "sktime",
        "pypots",
        "tensorflow",  # summary stuff for logging
        "regex",
        "cloudpickle",
        "tsfresh",
        # https://github.com/alan-turing-institute/sktime/issues/1478
        "statsmodels==0.12.1",  # https://github.com/alan-turing-institute/sktime/issues/1478#issuecomment-932816360
        # 0.13.2 will play nice with IndexInt64 Error.
        "hypothesis",
        "pyampute",
        "category_encoders",
    ],
    # ray for tuning, guild for tracking experiments, or notebook figures
    # click version conflict between newest black and guildai
    extras_require={
        "dev": [
            # https://github.com/Lightning-AI/lightning/discussions/11926
            "ray[tune]",
            "guildai",
            "plotly",
            "nbformat",
            "flake8",
            "mypy",
            "black",
        ]
    },
)

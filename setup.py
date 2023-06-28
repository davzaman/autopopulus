#!/usr/bin/env python
from distutils.core import setup

setup(
    name="autopopulus",
    version="2.0",
    description="Imputation suite that uses autoencoders for imputation.",
    author="Davina Zamanzadeh",
    author_email="davzaman@gmail.com",
    packages=["autopopulus"],
    install_requires=[
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
        "pyampute",
        "regex",
        "cloudpickle",
        "tsfresh",
        "statsmodels",
        "hypothesis",
        "pyampute",
        "category_encoders",
        # https://github.com/Lightning-AI/lightning/discussions/11926
        "ray[tune,default]",
    ],
    # ray for tuning, guild for tracking experiments, or notebook figures
    # click version conflict between newest black and guildai
    extras_require={
        "dev": [
            "guildai",
            "plotly",
            "nbstripout",
            "flake8",
            "mypy",
            "black",
            "snakeviz",
            "filprofiler",
            "aim",
            "mlflow",
            "pip-chill",
        ]
    },
)

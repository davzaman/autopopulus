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
        "pytorch-lightning",
        "pandas>=1.3.0",  # for multicol explode
        "torch>=1.8",  # nan_to_num
        "tensorboardX",
        "xgboost",
        "rich",  # nice stack traces + printing
        "miceforest",
        "torchmetrics",
        "numpy",
        "scikit-learn",
        "orange3",
        "tqdm",
        "pyarrow",  # if using feather preproc
        "fastparquet",
        "keras",  # if using keras for dnn
        "python>=3.5",  # abstract class / other python abilities
        "sktime",
        "pypots",
        "tensorflow",  # summary stuff for logging
        "regex",
        "cloudpickle",
    ],
)

# Autopopulus
Autopopulus  is a framework that enables the design, profiling, evaluation, and selection of various autoencoder architectures for efficient imputation on large datasets.
## Imputation
To use Autopopulus for imputation, we provide the `AEImputer` class, following on the `fit()`/`transform()` paradigm from sklearn.
`AEImputer` wraps an inner `AEDitto` object, a [Pytorch Lightning](https://www.pytorchlightning.ai/index.html) `LightningModule` that implements a flexible autoencoder.
The `AEImputer` expects data in the form of a `CommonDataModule` object, which at minimum requires a user define and pass a datasetloader.
Users may define their own dataset class that extends the `AbstractDatasetLoader` class, or they may use the `SimpleDatasetLoader` for convenience.
Each dataset loader requires the data (in the form of a [pandas](https://pandas.pydata.org/docs/index.html) `Dataframe`), the label data (in the form of a pandas `Series`) or the name of the label column, a list of continuous columns, a list of categorical columns, and one-hot encoding prefixes.
Optionally, if the data are already split into train, val, and test, `SimpleDatasetLoader` allows to indicate which column assigns each sample to a split.

Both `CommonDataModule` and `AEImputer` can be partially initialized using command-line arguments for convenience.
For more examples, explore `autopopulus/task_logic/ae_imputation.py`.
First initialize a `CommonDataModule` object for your dataset and call `setup()` on it.
After initializing `AEImputer` with the settings you like, to train the imputer call `aeimputer_object.fit(data_object)`.
If you choose to instead tune, call `aeimputer_object.tune(*args)`, refer to the function signature for more.
To impute with the imputer call `aeimputer_object.transform(dataset)`, where the dataset is expected to be a numpy array or a pandas dataframe.

## Comparison and Model Selection
Autopopulus consists of a 3-step pipeline for experimental workflow:
1. Imputation via `impute.py`.
2. Imputation evaluation via `evaluate.py`, which relies on the `impute.py` step upstream.
3. Downstream predictive evaluation via `predict.py`, which relies on the `impute.py` step upstream.

To run one-off experiments, you may create scripts that passes in command-line args, or optionally define an `options.yml` file at the root of the project.
Additionally, we have scripts in the `dev/experiment_running/` directory  to help execute runs, particularly the `RunManager` class and `experiment_track_all_imputers.py`.

Autopopulus currently supports 3 experiment trackers: [guildai](https://guild.ai/)/[Tensorboard](https://www.tensorflow.org/tensorboard), [Aim](https://aimstack.io/), and [MLflow](https://mlflow.org/).
To choose your experiment tracker, change the `LOGGER_TYPE` global variable in `autopopulus/utils.log_utils.py`.
In future versions we will make this parameter of the `AEImputer` class.
By default we use MLflow and will automatically serialize the trained imputer model and the imputed data for all data splits.

# Citation
Autopopulus was initially introduced in our earlier paper:
```
@inproceedings{zamanzadehAutopopulusNovelFramework2021a,
  title = {Autopopulus: {{A Novel Framework}} for {{Autoencoder Imputation}} on {{Large Clinical Datasets}}},
  shorttitle = {Autopopulus},
  booktitle = {2021 43rd {{Annual International Conference}} of the {{IEEE Engineering}} in {{Medicine}} \& {{Biology Society}} ({{EMBC}})},
  author = {Zamanzadeh, Davina J. and Petousis, Panayiotis and Davis, Tyler A. and Nicholas, Susanne B. and Norris, Keith C. and Tuttle, Katherine R. and Bui, Alex A. T. and Sarrafzadeh, Majid},
  year = {2021},
  month = nov,
  pages = {2303--2309},
  issn = {2694-0604},
}
```

[![DOI](https://zenodo.org/badge/362901433.svg)](https://zenodo.org/badge/latestdoi/362901433)
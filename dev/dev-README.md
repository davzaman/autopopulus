# Project Structure
- `options.yml`: argument values for running autopopulus in general (e.g., data paths to datasets, and other arguments to run imputation).
  When not running with guild, all ArgParse defaults will be overriden with values provided here, with command-line arugments at highest precedence.
- `guild.yml`: config for running experiments via guild for experiment tracking.
  Note that any flags defined here need to be passed in on calling `guild run ...`, the `options.yml` values will not be filled in, only the ArgParse defaults defined in the code.
- `scripts/`: directory of helpful scripts for running experiments and visualizing results.
- `notebooks/`: directory of jupyter notebooks for experimental or adhoc scripts
- `autopopulus/`: root directory for all the source code
  - `impute.py`: main logic for training an imputer/running imputation. Serialized imputed output.
  - `predict.py`: Loads imputed pickled data and runs predictions on them.
  - `evaluate.py`: Loads trained imputer (autoencoders only) and runs it on the test dataset.
  - `data/`: data wrangling procedures (i.e. transforms) and data loading for pytorch modules.
  - `datasets/`: directory of data loading procedures for particular datasets.
  - `models/`: directory of defining model architectures (both imputation models and predictive models that aren't prepackaged) and their training procedures.
  - `task_logic/`: directory defining imputation and prediction task logic.
  - `test/`: directory of tests for autopopulus
  - `utils/`: directory of genral utils (logging, discretization,  profiling-one-off-script, etc).

## Testing
The tests use `hypothesis`, which can be finicky.
If it cannot satisfy asssumptions, just try running that individual test again.
Maybe even try running another test and then run that test, sometimes it refreshes the cache and you end up with different inputs that might crash your test.

We also use `coverage` for test coverage.
Run it: `coverage run -m unittest discover`
Get report: `coverage report` or `coverage html`. Add `-i` to ignore errors.

## Dev Structure
These are files I can use to run experiments/etc without them being tracked as part of the projecct.
Run all the scripts here from the root of the project, not in `/dev/`. 

# Artifacts
- `profiling-results`: directory containing output of profilers after running `profile_autoencoder.py`.
- `serialized_models`: directory containing serialized models, pickled imputed data, and test_dataloaders to do downstream predictions and evaluate autoencoders after training with `evaluate.py`.
- `tune_results`: directory containing output of logging during tuning. Might be empty if tuning as part of automatic model selection.
- `data_description/ae_type_name/predictor_model`: tensorboard logs of predictive preformance under a certain data (full, amputed, etc), aeutoencoder type (ap_new, etc), and predictor model (lr, rf).
- `data_description/ae_type_name/lightning_logs`: tensorboard logs of training and evaluation metrics of autoencoder under certain data.

## Data Loading
For each new dataset added/supported write a data loader class in `data/`.
The class should extend `AbstractDatasetLoader`. Look at the definition of that class to know how to implement the data loader (implement all `@abstractmethod`s and `abstract_attribute()`s). They will also be able to initialized via CLI arguments with `from_argparse_args` if you wish.

You can also just use `SimpleDatasetLoader` for simple datasets (also can be initialized from CLI args).
```python
SimpleDatasetLoader(
    df,
    label="label_name",
    continuous_cols=df.filter(regex=continuous_cols_regex).columns,
    categorical_cols=df.filter(regex=categorical_cols_regex).columns,
    onehot_prefixes=["abc"],
)
```

# Environment
Install the env with conda or mamba. With mamba:
```shell
env_name=""
env_file="env.yml"
mamba create -n $env_name --no-default-packages -y
mamba activate $env_name
mamba env update -n $env_name -f $env_file
```

I might have to downgrade torch because of old CUDA. 
[Installation Instructions](https://pytorch.org/get-started/previous-versions/)
`mamba install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=10.2 -c pytorch`


You might see `AttributeError: module 'typing' has no attribute '_ClassVar'`, in which case [`pip uninstall dataclasses`](https://github.com/ray-project/tune-sklearn/issues/181#issuecomment-782598003).


## Technology
We use pytorch-lightning. Important tips and tricks [here](https://lightning.ai/docs/pytorch/stable/advanced/speed.html).
## Ray[Tune]
If using Ray[Tune], to monitor the tuning experiments with a dashboard you can use `tune.init(include_dashboard=True)`.
However, if you are running into issues, you can check the logs at `cat /tmp/ray/session_latest/logs/dashboard.log` or `cat /tmp/ray/session_latest/logs/dashboard_agent.log`.
It turns out you need `ray-default` in order to use the dashboard in addition to `ray-tune`.
Ray Tune is on pip and conda-forge. You can check if conda-forge has the most up-to-date versions [here](https://github.com/conda-forge/ray-packages-feedstock).
Installation instructions from ray [here](https://docs.ray.io/en/latest/ray-overview/installation.html#installing-from-conda-forge).

# Includes:
- A .gitignore file from this [source](https://raw.githubusercontent.com/github/gitignore/master/Python.gitignore)
- The nbstripout library that removes any jupyter notebook outputs to be avoided from versioning.
  - Follow instructions on [official repo site](https://github.com/kynan/nbstripout)
  - `pip install --upgrade nbstripout`
  - Navigate to repository directory and run on the terminal `nbstripout --install`
  - Check the created filters: `nbstripout --status`

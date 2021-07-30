# Autopopulus
An implementation of autoencoder imputation from our paper "Autopopulus: A Novel Framework for Autoencoder Imputation on Large Clinical Datasets".

# Summary
Usage of the `AEImputer` class is similar to sklearn's imputers.
Both `CommonDataModule` and `AEImputer` can be partially initialized using command-line arguments.
For more examples, explore `task_logic/ae_experiments.py`.

First initialize a `CommonDataModule` object for your dataset and call `setup()` on it.
After initializing `AEImputer` with the settings you like, to train the imputer call `aeimputer_object.fit(data_object)`.
To impute with the imputer call `aeimputer_object.transform(dataset)`, where the dataset is expected to be a numpy array or a pandas dataframe.

To run experiments, edit hyperparameters in `guild.yml` and then  run `python imputer.py`.
To tune hyperparameters edit the config ranges in `tuner.py` and then call `run_tune()` in your routine.

# References
All methods are implemented natively, but for reference:
- Pytorch implemention of MIDA available by [Gondara L., et. al.](https://github.com/Harry24k/MIDA-pytorch).
- Implementation of DAPS by [Beaulieu-Jones et. al.](https://github.com/greenelab/DAPS).
- Implementation of IFAC-VAE by [McCoy et. al.](https://github.com/ProcessMonitoringStellenboschUniversity/IFAC-VAE-Imputation).

## Collaborators
Parts of code either additionally reviewed or provided by [Panayiotis Petousis](https://github.com/panas89) and [Tyler Davis](https://github.com/TylerADavis).

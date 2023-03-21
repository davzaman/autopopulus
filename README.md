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

# Citation
If you use this framework in your work please cite it
```
@INPROCEEDINGS{9630135,
  author={Zamanzadeh, Davina J. and Petousis, Panayiotis and Davis, Tyler A. and Nicholas, Susanne B. and Norris, Keith C. and Tuttle, Katherine R. and Bui, Alex A. T. and Sarrafzadeh, Majid},
  booktitle={2021 43rd Annual International Conference of the IEEE Engineering in Medicine & Biology Society (EMBC)}, 
  title={Autopopulus: A Novel Framework for Autoencoder Imputation on Large Clinical Datasets}, 
  year={2021},
  volume={},
  number={},
  pages={2303-2309},
  doi={10.1109/EMBC46164.2021.9630135}}
```

from typing import Dict
from autopopulus.data.dataset_classes import AbstractDatasetLoader
from autopopulus.datasets.ckd import CureCKDDataLoader
from autopopulus.datasets.crrt import CrrtDataLoader

DATA_LOADERS: Dict[str, AbstractDatasetLoader] = {
    "cure_ckd": CureCKDDataLoader,
    "crrt": CrrtDataLoader,
}

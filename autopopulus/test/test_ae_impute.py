import unittest
from argparse import Namespace
from shutil import rmtree
from typing import Dict, List, Optional, Tuple, Union
from unittest.mock import ANY, call, patch

import numpy as np
from numpy.random import default_rng
from numpy.testing import assert_array_equal
import pandas as pd


from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st
from hypothesis.extra.pandas import data_frames


from pytorch_lightning.loggers.logger import DummyLogger

import torch
import torch.nn as nn
from torch import Generator, isnan
from torch import long as torch_long
from torch import nan_to_num, rand, randn, tensor
from torch.autograd import Variable
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.loss import BCEWithLogitsLoss, MSELoss
from torch.testing import assert_allclose

from autopopulus.data import CommonDataModule
from autopopulus.data.constants import PAD_VALUE
from autopopulus.data.transforms import list_to_tensor
from autopopulus.models.ae import COL_IDXS_BY_TYPE_FORMAT, AEDitto
from autopopulus.models.ap import AEImputer
from autopopulus.models.dnn import ResetSeed
from models.torch_model_utils import (
    BatchSwapNoise,
    BinColumnThreshold,
    CtnCatLoss,
    OnehotColumnThreshold,
    ReconstructionKLDivergenceLoss,
)
from autopopulus.test.common_mock_data import (
    X,
    col_idxs_by_type,
    columns,
    hypothesis,
    seed,
    y,
    discretization,
)
from autopopulus.test.utils import (
    build_onehot_from_hypothesis,
    get_dataset_loader,
    mock_disc_data,
)
from autopopulus.utils.log_utils import (
    IMPUTE_METRIC_TAG_FORMAT,
    get_serialized_model_path,
)
from autopopulus.data.dataset_classes import SimpleDatasetLoader

seed = 0
basic_imputer_args = {
    "hidden_layers": [3, 2, 3],
    "learning_rate": 0.1,
    "seed": seed,
}
layer_dims = [6, 3, 2, 3, 6]
EPSILON = 1e-10


def get_data_args(
    col_idxs_set_empty: Optional[Dict] = None,
) -> Dict[str, Union[str, Dict]]:
    res = {
        "data_feature_space": "original",
        "feature_map": "none",
        "nfeatures": {"original": len(columns["columns"])},
        "col_idxs_by_type": {
            "original": {
                k: tensor(v, dtype=torch_long)
                for k, v in col_idxs_by_type["original"].items()
            }
        },
        "columns": {"original": columns["columns"]},
        "discretizations": None,
        "inverse_target_encode_map": None,
    }
    if col_idxs_set_empty is not None:
        for key in col_idxs_set_empty:
            res["col_idxs_by_type"]["original"][key] = []
    return res


def mock_training_step(self, batch, split):
    # data on gpu
    assert batch[self.hparams.data_feature_space]["data"].is_cuda
    # model on gpu
    assert next(self.encoder.parameters()).is_cuda
    assert next(self.decoder.parameters()).is_cuda
    if self.hparams.variational:
        assert next(self.fc_mu.parameters()).is_cuda
        assert next(self.fc_var.parameters()).is_cuda
    # metric on gpu
    for split_level_metrics in self.metrics["train_metrics"].values():
        for high_level_metrics in split_level_metrics.values():
            if "original" in high_level_metrics:
                high_level_metrics = high_level_metrics["original"]
            elif "mapped" in high_level_metrics:
                high_level_metrics = high_level_metrics["mapped"]
            for metric in high_level_metrics.values():
                assert metric.device == self.device
    return self.shared_step(batch, "train")[0]


class TestAEImputer(unittest.TestCase):
    def setUp(self) -> None:
        nsamples = len(X["X"])
        rng = default_rng(seed)
        y = pd.Series(rng.integers(0, 2, nsamples))  # random binary outcome
        self.data_settings = {
            "dataset_loader": get_dataset_loader(X["X"], y),
            "seed": seed,
            "val_test_size": 0.5,
            "test_size": 0.5,
            "batch_size": 2,
        }
        self.datamodule = CommonDataModule(**self.data_settings, scale=True)
        self.aeimp = AEImputer(
            **basic_imputer_args,
            replace_nan_with=0,
            max_epochs=3,
            num_gpus=0,
            early_stopping=False,  # our logging is mocked so we won't have metrics
        )

    @patch("autopopulus.data.dataset_classes.train_test_split")
    def test_logging(self, mock_split):
        mock_split.return_value = (X["X"].index, X["X"].index)
        with self.subTest("No Feature Map"):
            with patch("autopopulus.models.ae.LightningModule.log") as mock_log:
                self.aeimp.fit(self.datamodule)  # rerun to be inside mock/patch
                # this implicitly tests AEDitto:init_metrics()
                for split in ["train", "val"]:
                    mock_log.assert_any_call(
                        IMPUTE_METRIC_TAG_FORMAT.format(
                            name="loss",
                            feature_space="original",  # no feature map
                            filter_subgroup="all",
                            reduction="NA",
                            split=split,
                        ),
                        ANY,
                        on_step=ANY,
                        on_epoch=ANY,
                        prog_bar=ANY,
                        logger=ANY,
                        rank_zero_only=ANY,
                    )
                for split in ["train", "val"]:
                    for filter_subgroup in ["all", "missingonly"]:
                        for reduction in ["CW", "EW"]:
                            for metric in ["RMSE", "MAAPE"]:
                                mock_log.assert_any_call(
                                    IMPUTE_METRIC_TAG_FORMAT.format(
                                        name=metric,
                                        feature_space="original",
                                        filter_subgroup=filter_subgroup,
                                        reduction=reduction,
                                        split=split,
                                    ),
                                    ANY,
                                    on_step=ANY,
                                    on_epoch=ANY,
                                    prog_bar=ANY,
                                    logger=ANY,
                                    rank_zero_only=ANY,
                                    # *[ANY for i in range(6)],
                                )
        with self.subTest("With Feature Map"):
            with patch("autopopulus.models.ae.LightningModule.log") as mock_log:
                datamodule = CommonDataModule(
                    **self.data_settings,
                    feature_map="discretize_continuous",
                    uniform_prob=True,
                    scale=True,
                )
                self.aeimp.ae_kwargs["lossn"] = "BCE"
                self.aeimp.fit(datamodule)
                # this implicitly tests AEDitto:init_metrics()
                for split in ["train", "val"]:
                    mock_log.assert_any_call(
                        IMPUTE_METRIC_TAG_FORMAT.format(
                            name="loss",
                            feature_space="mapped",  # loss data isn't inverted
                            filter_subgroup="all",
                            reduction="NA",
                            split=split,
                        ),
                        ANY,
                        on_step=ANY,
                        on_epoch=ANY,
                        prog_bar=ANY,
                        logger=ANY,
                        rank_zero_only=ANY,
                    )
                for split in ["train", "val"]:
                    for filter_subgroup in ["all", "missingonly"]:
                        for reduction in ["CW", "EW"]:
                            feature_spaces = ["original"]
                            metrics = ["RMSE", "MAAPE"]
                            if reduction == "CW":
                                feature_spaces.append("mapped")
                                metrics.append("Accuracy")
                            for feature_space in feature_spaces:
                                # disc -> accuracy
                                for metric in metrics:
                                    mock_log.assert_any_call(
                                        IMPUTE_METRIC_TAG_FORMAT.format(
                                            name=metric,
                                            feature_space=feature_space,
                                            filter_subgroup=filter_subgroup,
                                            reduction=reduction,
                                            split=split,
                                        ),
                                        ANY,
                                        on_step=ANY,
                                        on_epoch=ANY,
                                        prog_bar=ANY,
                                        logger=ANY,
                                        rank_zero_only=ANY,
                                        # *[ANY for i in range(6)],
                                    )

    @patch("autopopulus.data.dataset_classes.train_test_split")
    def test_basic(self, mock_split):
        mock_split.return_value = (X["X"].index, X["X"].index)
        self.aeimp.fit(self.datamodule)
        train_dataloader = self.datamodule.train_dataloader()
        with self.subTest("Transform Function"):
            df = self.aeimp.transform(train_dataloader)

            self.assertEqual(df.isna().sum().sum(), 0)
            # the dataset is transformed so the values shouldn't be the same but the shape, columns, index should
            assert_array_equal(df.index, X["X"].index)
            assert_array_equal(df.columns, X["X"].columns)
            self.assertEqual(df.shape, X["X"].shape)

        with self.subTest("Load checkpoint"):
            other_aeimp = AEImputer.from_checkpoint(
                Namespace(
                    **basic_imputer_args,
                    replace_nan_with=0,
                    max_epochs=3,
                    num_gpus=0,
                    early_checkpointing=False,
                    method="whatever",
                ),
                get_serialized_model_path(f"AEDitto_STATIC", "pt"),
            )
            other_df = other_aeimp.transform(train_dataloader)
            pd.testing.assert_frame_equal(other_df, df)
            self.assertEqual(
                other_aeimp.ae.hparams.keys(), self.aeimp.ae.hparams.keys()
            )
            for k in other_aeimp.ae.hparams:
                item = other_aeimp.ae.hparams[k]
                if isinstance(item, dict):
                    if isinstance(list(item.values())[0], pd.Index):
                        self.assertEqual(item.keys(), self.aeimp.ae.hparams[k].keys())
                        for inner_k in item:
                            pd.testing.assert_index_equal(
                                item[inner_k], self.aeimp.ae.hparams[k][inner_k]
                            )
                    else:
                        np.testing.assert_equal(item, self.aeimp.ae.hparams[k])
                elif isinstance(item, list):
                    np.testing.assert_equal(item, self.aeimp.ae.hparams[k])
                else:
                    np.testing.assert_equal(item, self.aeimp.ae.hparams[k])
            self.assertDictEqual(
                other_aeimp.ae.hidden_and_cell_state,
                self.aeimp.ae.hidden_and_cell_state,
            )
            self.assertEqual(
                str(other_aeimp.ae.state_dict()), str(self.aeimp.ae.state_dict())
            )
        rmtree("whatever")

    @torch.no_grad()
    @patch("autopopulus.data.dataset_classes.train_test_split")
    def test_data_transforms(self, mock_split):
        """Can't set scale because I can't guarantee how the values will come back."""
        onehot_df = X["X"]
        mock_split.return_value = (onehot_df.index, onehot_df.index)

        datamodule = CommonDataModule(**self.data_settings)
        aeimp = AEImputer(
            **basic_imputer_args,
            replace_nan_with=0,
            max_epochs=3,
            num_gpus=0,
        )
        aeimp.fit(datamodule)
        train_dataloader = datamodule.train_dataloader()
        res = aeimp.transform(train_dataloader)
        pd.testing.assert_frame_equal(  # ignore nans in comparison
            res.where(~onehot_df.isna(), onehot_df), onehot_df, check_dtype=False
        )
        with self.subTest("discretize_continuous"):
            datamodule = CommonDataModule(
                **self.data_settings,
                feature_map="discretize_continuous",
                uniform_prob=True,
            )
            aeimp.ae_kwargs["lossn"] = "BCE"
            aeimp.fit(datamodule)
            train_dataloader = datamodule.train_dataloader()
            res = aeimp.transform(train_dataloader)
            pd.testing.assert_frame_equal(  # ignore nans in comparison
                res.where(~onehot_df.isna(), onehot_df), onehot_df, check_dtype=False
            )

        with self.subTest("target_encode_categorical"):
            datamodule = CommonDataModule(
                **self.data_settings,
                feature_map="target_encode_categorical",
            )
            aeimp.ae_kwargs["lossn"] = "MSE"
            aeimp.fit(datamodule)
            train_dataloader = datamodule.train_dataloader()
            res = aeimp.transform(train_dataloader)
            pd.testing.assert_frame_equal(  # ignore nans in comparison
                res.where(~onehot_df.isna(), onehot_df), onehot_df, check_dtype=False
            )

    @torch.no_grad()
    @unittest.skipUnless(torch.cuda.is_available(), "No GPU was detected")
    @patch.object(AEDitto, "training_step", mock_training_step)
    @patch("autopopulus.data.dataset_classes.train_test_split")
    def test_device(self, mock_split):
        mock_split.return_value = (X["X"].index, X["X"].index)
        aeimp = AEImputer(
            **basic_imputer_args,
            replace_nan_with=0,
            max_epochs=3,
            num_gpus=1,  # ensure num_gpus is > 0 to check for cuda
            early_stopping=False,  # our logging is mocked so we won't have metrics
        )
        aeimp.fit(self.datamodule)
        with self.subTest("discretize_continuous"):
            datamodule = CommonDataModule(
                **self.data_settings,
                feature_map="discretize_continuous",
                uniform_prob=True,
                scale=True,
            )
            aeimp.ae_kwargs["lossn"] = "BCE"
            aeimp.fit(datamodule)
        with self.subTest("target_encode_categorical"):
            datamodule = CommonDataModule(
                **self.data_settings,
                feature_map="target_encode_categorical",
                scale=True,
            )
            aeimp.ae_kwargs["lossn"] = "MSE"
            aeimp.fit(datamodule)

            with self.subTest("VAE"):  # has additional layers, need ctn featuers
                aeimp = AEImputer(
                    **basic_imputer_args,
                    variational=True,
                    replace_nan_with=0,
                    max_epochs=3,
                    num_gpus=1,  # check data, model, and metrics go on the gpu
                )
                aeimp.ae_kwargs["lossn"] = "MSE"
                # Turn off logging for testing
                aeimp.fit(datamodule)

    @patch(
        "autopopulus.data.mdl_discretization.MDLDiscretizer.get_discretized_MDL_data"
    )
    @patch("autopopulus.data.mdl_discretization.MDLDiscretizer.bin_ranges_as_tuples")
    @patch("autopopulus.data.dataset_classes.train_test_split")
    def test_ae_init(self, mock_split, mock_disc_cuts, mock_MDL):
        """
        Let AEDitto take care of testing for model creation testing.
        Here we're concerned with making sure what AEditto receives from AEImputer/the data is correct.
        Looking at: feature_map_inversion, col_idxs_by_type, and metrics.
        """
        mock_split.return_value = (X["X"].index, X["X"].index)
        self.aeimp.fit(self.datamodule)
        self.assertIsNone(self.aeimp.ae.feature_map_inversion)
        self._test_set_col_idxs_by_type(
            self.aeimp.ae,
            [
                ("original", "binary"),
                ("original", "onehot"),
                ("original", "continuous"),
            ],
        )
        metrics = self.aeimp.ae.metrics
        self.assertEqual(
            list(metrics.keys()), ["train_metrics", "val_metrics", "test_metrics"]
        )
        for split_moduledict in metrics.values():
            self.assertEqual(list(split_moduledict.keys()), ["all", "missingonly"])
            for subgroup_moduledict in split_moduledict.values():
                self.assertEqual(list(subgroup_moduledict.keys()), ["CW", "EW"])
                for reduction, reduction_moduledict in subgroup_moduledict.items():
                    if reduction == "CW":
                        self.assertEqual(
                            list(reduction_moduledict.keys()), ["original"]
                        )
                        for leaf_metrics in reduction_moduledict.values():
                            self.assertEqual(
                                list(leaf_metrics.keys()), ["RMSE", "MAAPE"]
                            )
                    else:
                        self.assertEqual(
                            list(reduction_moduledict.keys()), ["RMSE", "MAAPE"]
                        )

        with self.subTest("discretize_continuous"):
            mock_disc_cuts.return_value = discretization["cuts"]
            mock_disc_data(mock_MDL, X["disc"], y)
            datamodule = CommonDataModule(
                **self.data_settings,
                feature_map="discretize_continuous",
            )
            self.aeimp.ae_kwargs["lossn"] = "BCE"
            self.aeimp.fit(datamodule)
            self.assertIsNotNone(self.aeimp.ae.feature_map_inversion)
            # f^-1(f(x)) = x, we should get the original data back
            train_dataloader = next(iter(datamodule.train_dataloader()))
            assert_allclose(  # cat cols should match values and order
                self.aeimp.ae.feature_map_inversion(train_dataloader["mapped"]["data"])[
                    :, col_idxs_by_type["original"]["categorical"]
                ],
                train_dataloader["original"]["data"][
                    :, col_idxs_by_type["original"]["categorical"]
                ],
                equal_nan=True,
            )
            self._test_set_col_idxs_by_type(
                self.aeimp.ae,
                [
                    (data_feature_space, feature_type)
                    for data_feature_space in ["original", "mapped"]
                    for feature_type in ["binary", "onehot", "continuous"]
                ],
            )
            metrics = self.aeimp.ae.metrics
            self.assertEqual(
                list(metrics.keys()), ["train_metrics", "val_metrics", "test_metrics"]
            )
            for split_moduledict in metrics.values():
                self.assertEqual(list(split_moduledict.keys()), ["all", "missingonly"])
                for subgroup_moduledict in split_moduledict.values():
                    self.assertEqual(list(subgroup_moduledict.keys()), ["CW", "EW"])
                    for reduction, reduction_moduledict in subgroup_moduledict.items():
                        if reduction == "CW":
                            self.assertEqual(
                                list(reduction_moduledict.keys()),
                                ["original", "mapped"],
                            )
                            for leaf_metrics in reduction_moduledict.values():
                                self.assertEqual(
                                    list(leaf_metrics.keys()),
                                    ["RMSE", "MAAPE", "Accuracy"],
                                )
                        else:
                            self.assertEqual(
                                list(reduction_moduledict.keys()), ["RMSE", "MAAPE"]
                            )
        with self.subTest("target_encode_categorical"):
            datamodule = CommonDataModule(
                **self.data_settings,
                feature_map="target_encode_categorical",
            )
            self.aeimp.ae_kwargs["lossn"] = "MSE"
            self.aeimp.fit(datamodule)
            self.assertIsNotNone(self.aeimp.ae.feature_map_inversion)
            train_dataloader = next(iter(datamodule.train_dataloader()))
            # we can't compare the mapped columns bc the inversion is inexact
            assert_allclose(  # ctn cols should be equal and in order
                self.aeimp.ae.feature_map_inversion(train_dataloader["mapped"]["data"])[
                    :, col_idxs_by_type["original"]["continuous"]
                ],
                train_dataloader["original"]["data"][
                    :, col_idxs_by_type["original"]["continuous"]
                ],
                equal_nan=True,
            )
            self._test_set_col_idxs_by_type(
                self.aeimp.ae,
                [
                    (data_feature_space, feature_type)
                    for data_feature_space in ["original", "mapped"]
                    for feature_type in ["binary", "onehot", "continuous"]
                ],
            )
            metrics = self.aeimp.ae.metrics
            self.assertEqual(
                list(metrics.keys()), ["train_metrics", "val_metrics", "test_metrics"]
            )
            for split_moduledict in metrics.values():
                self.assertEqual(list(split_moduledict.keys()), ["all", "missingonly"])
                for subgroup_moduledict in split_moduledict.values():
                    self.assertEqual(list(subgroup_moduledict.keys()), ["CW", "EW"])
                    for reduction, reduction_moduledict in subgroup_moduledict.items():
                        if reduction == "CW":
                            self.assertEqual(
                                list(reduction_moduledict.keys()),
                                ["original", "mapped"],
                            )
                            for leaf_metrics in reduction_moduledict.values():
                                self.assertEqual(  # no accuracy bc target_encode
                                    list(leaf_metrics.keys()), ["RMSE", "MAAPE"]
                                )
                        else:
                            self.assertEqual(
                                list(reduction_moduledict.keys()), ["RMSE", "MAAPE"]
                            )

    def _test_set_col_idxs_by_type(
        self, model: AEDitto, feature_space_and_type_pairs: List[Tuple[str, str]]
    ):
        for data_feature_space, feature_type in feature_space_and_type_pairs:
            self.assertTrue(
                hasattr(
                    model,
                    COL_IDXS_BY_TYPE_FORMAT.format(
                        data_feature_space=data_feature_space, feature_type=feature_type
                    ),
                )
            )


class TestAEDitto(unittest.TestCase):
    @patch("autopopulus.models.ae.onehot_column_threshold")
    @patch("autopopulus.models.ae.binary_column_threshold")
    def test_get_imputed_tensor_from_model_output(
        self,
        mock_binary_column_threshold,
        mock_onehot_column_threshold,
    ):
        ae = AEDitto(
            **basic_imputer_args,
            **get_data_args(),
            lossn="BCE",
        )
        ae.setup("fit")
        fill_val = tensor(-5000)

        data = tensor(X["X"].values)
        true = tensor(X["nomissing"].values)
        non_missing_mask = tensor(~X["X"].isna().values).bool()
        # make it wrong in all the observed places and the same fill value for the missing ones
        reconstruct_batch = (data * -1).where(non_missing_mask, fill_val)
        # Test these separately, they're noops here
        mock_binary_column_threshold.return_value = reconstruct_batch
        mock_onehot_column_threshold.return_value = reconstruct_batch

        inputs = (data, reconstruct_batch, true, non_missing_mask)
        cloned_inputs = (x.clone() for x in inputs)
        (
            imputed,
            res_ground_truth,
            res_non_missing_mask,
        ) = ae.get_imputed_tensor_from_model_output(
            *inputs,
            original_data=None,
            original_ground_truth=None,
            data_feature_space="original",
        )
        # observed values should be correct, missing values should be the fill value
        assert_allclose(imputed, nan_to_num(data, fill_val))
        # should come out the same, no changes
        assert_allclose(res_ground_truth, true)
        # should come out the same, no changes
        assert_allclose(res_non_missing_mask, non_missing_mask)
        # make sure the originals are not mutated
        for tens, cloned_tens in zip(inputs, cloned_inputs):
            assert_allclose(tens, cloned_tens)

        with self.subTest("NaNs in Ground Truth"):
            inputs = (data, reconstruct_batch, data, non_missing_mask)
            cloned_inputs = (x.clone() for x in inputs)
            (
                imputed,
                res_ground_truth,
                res_non_missing_mask,
            ) = ae.get_imputed_tensor_from_model_output(
                *inputs,
                original_data=None,
                original_ground_truth=None,
                data_feature_space="original",
            )
            # observed values should be correct, missing values should be the fill value
            assert_allclose(imputed, nan_to_num(data, fill_val))
            # ground truth should have the predicted values filled in
            assert_allclose(res_ground_truth, nan_to_num(data, fill_val))
            # should come out the same, no changes
            assert_allclose(res_non_missing_mask, non_missing_mask)
            # make sure the originals are not mutated
            for tens, cloned_tens in zip(inputs, cloned_inputs):
                assert_allclose(tens, cloned_tens)

        with self.subTest("Feature Mapped Data"):
            data_mapped = tensor(X["target_encoded"].values)
            data_original = tensor(X["X"].values)
            non_missing_mask_mapped = ~(isnan(data_mapped))
            non_missing_mask_original = ~(isnan(data_original))
            # make it wrong in all the observed places and the same fill value for the missing ones
            reconstruct_batch_mapped = (data_mapped * -1).where(
                non_missing_mask_mapped, fill_val
            )
            reconstruct_batch_original = (data_original * -1).where(
                non_missing_mask_original, fill_val
            )
            ground_truth_mapped = tensor(X["target_encoded_true"].values)
            ground_truth_original = tensor(X["nomissing"].values)

            # we simulate the mapping
            ae.feature_map = "target_encode_categorical"
            ae.feature_map_inversion = lambda x: reconstruct_batch_original
            # Test these separately, they're noops here
            mock_binary_column_threshold.return_value = reconstruct_batch_original
            mock_onehot_column_threshold.return_value = reconstruct_batch_original

            inputs = (
                data_mapped,
                reconstruct_batch_mapped,
                non_missing_mask_mapped,
                ground_truth_mapped,
                data_original,
                ground_truth_original,
            )
            cloned_inputs = (x.clone() for x in inputs)
            (
                imputed,
                res_ground_truth,
                res_non_missing_mask,
            ) = ae.get_imputed_tensor_from_model_output(
                *inputs,
                data_feature_space="original",  # even though it's mapped we want it in original
            )
            # observed values should be correct, missing values should be the fill value
            assert_allclose(imputed, nan_to_num(data_original, fill_val))
            assert_allclose(res_ground_truth, ground_truth_original)
            assert_allclose(res_non_missing_mask, non_missing_mask_original)
            # make sure the originals are not mutated
            for tens, cloned_tens in zip(inputs, cloned_inputs):
                assert_allclose(tens, cloned_tens)

    def test_idxs_to_tensor(self):
        with self.subTest("List"):
            l = [1, 2]
            assert_allclose(list_to_tensor(l), tensor(l))

        with self.subTest("List of List (No Padding)"):
            l = [[1, 2], [3, 4]]
            assert_allclose(list_to_tensor(l), tensor(l))

        with self.subTest("List of List (Padding)"):
            l = [[1, 2], [3]]
            assert_allclose(list_to_tensor(l), tensor([[1, 2], [3, PAD_VALUE]]))

    def test_vae(self):
        ae = AEDitto(
            **basic_imputer_args,
            **get_data_args(
                col_idxs_set_empty=["categorical", "binary_vars", "onehot"],
            ),
            variational=True,
            lossn="MSE",
        )

        ae.setup("fit")
        self.assertTrue(hasattr(ae, "fc_mu"))
        self.assertTrue(hasattr(ae, "fc_var"))
        self.assertIsInstance(ae.loss, ReconstructionKLDivergenceLoss)

        encoder = nn.ModuleList(
            [
                nn.Linear(6, 3),
                nn.ReLU(inplace=True),
            ]
        )
        mu_var = nn.Linear(3, 2)
        decoder = nn.ModuleList(
            [nn.Linear(2, 3), nn.ReLU(inplace=True), nn.Linear(3, 6)]
        )
        self.assertEqual(ae.encoder.__repr__(), encoder.__repr__())
        self.assertEqual(ae.decoder.__repr__(), decoder.__repr__())
        self.assertEqual(ae.fc_mu.__repr__(), mu_var.__repr__())
        self.assertEqual(ae.fc_var.__repr__(), mu_var.__repr__())

        pytorch_total_params = sum(
            p.numel() for p in ae.parameters() if p.requires_grad
        )
        self.assertEqual(
            pytorch_total_params,
            2 * ((6 * 3) + (3 * 2)) + (3 * 2) + sum(layer_dims[1:]) + 2,
        )

    def test_basic(self):
        ae = AEDitto(
            **basic_imputer_args,
            **get_data_args(
                col_idxs_set_empty=["continuous", "categorical", "binary", "onehot"],
            ),
            lossn="BCE",
        )
        ae.setup("fit")
        with self.subTest("Basic"):
            self.assertFalse(hasattr(ae, "fc_mu"))
            self.assertFalse(hasattr(ae, "fc_var"))

            self.assertEqual(ae.code_index, 2)
            self.assertEqual(ae.hparams.layer_dims, layer_dims)

            encoder = nn.ModuleList(
                [
                    nn.Linear(6, 3),
                    nn.ReLU(inplace=True),
                    nn.Linear(3, 2),
                    nn.ReLU(inplace=True),
                ]
            )
            decoder = nn.ModuleList(
                [
                    nn.Linear(2, 3),
                    nn.ReLU(inplace=True),
                    nn.Linear(3, 6),
                ]
            )
            self.assertEqual(ae.encoder.__repr__(), encoder.__repr__())
            self.assertEqual(ae.decoder.__repr__(), decoder.__repr__())

            pytorch_total_params = sum(
                p.numel() for p in ae.parameters() if p.requires_grad
            )
            self.assertEqual(
                pytorch_total_params, 2 * ((6 * 3) + (3 * 2)) + sum(layer_dims[1:])
            )

        # Fractional hidden layer
        with self.subTest("Fractional hidden layer"):
            new_settings = basic_imputer_args.copy()
            new_settings["hidden_layers"] = [0.6, 0.1, 0.05, 0.1, 0.6]
            ae = AEDitto(
                **new_settings,
                **get_data_args(),
            )
            ae.setup("fit")
            encoder = nn.ModuleList(
                [
                    nn.Linear(6, 4),
                    nn.ReLU(inplace=True),
                    nn.Linear(4, 1),
                    nn.ReLU(inplace=True),
                    nn.Linear(1, 1),
                    nn.ReLU(inplace=True),
                ]
            )
            decoder = nn.ModuleList(
                [
                    nn.Linear(1, 1),
                    nn.ReLU(inplace=True),
                    nn.Linear(1, 4),
                    nn.ReLU(inplace=True),
                    nn.Linear(4, 6),
                ]
            )
            self.assertEqual(ae.encoder.__repr__(), encoder.__repr__())
            self.assertEqual(ae.decoder.__repr__(), decoder.__repr__())

        # Loss Test
        with self.subTest("Loss"):
            ae = AEDitto(
                **basic_imputer_args,
                **get_data_args(),
                lossn="BCE",
            )
            ae.setup("fit")
            self.assertIsInstance(ae.loss, BCEWithLogitsLoss)

        # Dropout
        with self.subTest("Dropout"):
            ae = AEDitto(
                **basic_imputer_args,
                **get_data_args(),
                dropout=0.5,
            )
            ae.setup("fit")
            encoder = nn.ModuleList(
                [
                    nn.Linear(6, 3),
                    nn.ReLU(inplace=True),
                    ResetSeed(seed),
                    Dropout(0.5),
                    nn.Linear(3, 2),
                    nn.ReLU(inplace=True),
                    ResetSeed(seed),
                    Dropout(0.5),
                ]
            )
            decoder = nn.ModuleList(
                [
                    nn.Linear(2, 3),
                    nn.ReLU(inplace=True),
                    ResetSeed(seed),
                    Dropout(0.5),
                    nn.Linear(3, 6),
                ]
            )
            self.assertEqual(ae.encoder.__repr__(), encoder.__repr__())
            self.assertEqual(ae.decoder.__repr__(), decoder.__repr__())

    def test_longitudinal(self):
        ae = AEDitto(
            **basic_imputer_args,
            **get_data_args(),
            longitudinal=True,
        )
        ae.setup("fit")
        encoder = nn.ModuleList(
            [
                nn.LSTM(6, 3, batch_first=True),
                nn.ReLU(inplace=True),
                nn.LSTM(3, 2, batch_first=True),
                nn.ReLU(inplace=True),
            ]
        )
        decoder = nn.ModuleList(
            [
                nn.LSTM(2, 3, batch_first=True),
                nn.ReLU(inplace=True),
                nn.Linear(3, 6),
            ]
        )
        self.assertEqual(ae.encoder.__repr__(), encoder.__repr__())
        self.assertEqual(ae.decoder.__repr__(), decoder.__repr__())

        with self.subTest("Apply Layers"):
            X_long = rand(size=(6, 6, 6), generator=Generator().manual_seed(seed))
            code = ae.encode("train", X_long, tensor([6] * 6))
            self.assertEqual(ae.curr_rnn_depth, 1)

            ae.decode("train", code, tensor([6] * 6))
            self.assertEqual(ae.curr_rnn_depth, 0)

    def test_dae(self):
        with self.subTest("Dropout Corruption"):
            ae = AEDitto(
                **basic_imputer_args,
                **get_data_args(),
                dropout_corruption=0.5,
            )
            ae.setup("fit")
            encoder = nn.ModuleList(
                [
                    ResetSeed(seed),
                    Dropout(0.5),
                    nn.Linear(6, 3),
                    nn.ReLU(inplace=True),
                    nn.Linear(3, 2),
                    nn.ReLU(inplace=True),
                ]
            )
            decoder = nn.ModuleList(
                [
                    nn.Linear(2, 3),
                    nn.ReLU(inplace=True),
                    nn.Linear(3, 6),
                ]
            )
            self.assertEqual(ae.encoder.__repr__(), encoder.__repr__())
            self.assertEqual(ae.decoder.__repr__(), decoder.__repr__())

        with self.subTest("Batchswap Corruption"):
            ae = AEDitto(
                **basic_imputer_args,
                **get_data_args(),
                batchswap_corruption=0.5,
            )
            ae.setup("fit")
            encoder = nn.ModuleList(
                [
                    BatchSwapNoise(0.5),
                    nn.Linear(6, 3),
                    nn.ReLU(inplace=True),
                    nn.Linear(3, 2),
                    nn.ReLU(inplace=True),
                ]
            )
            decoder = nn.ModuleList(
                [
                    nn.Linear(2, 3),
                    nn.ReLU(inplace=True),
                    nn.Linear(3, 6),
                ]
            )
            self.assertEqual(ae.encoder.__repr__(), encoder.__repr__())
            self.assertEqual(ae.decoder.__repr__(), decoder.__repr__())

        with self.subTest("Dropout and Batchswap Corruption"):
            ae = AEDitto(
                **basic_imputer_args,
                **get_data_args(),
                dropout_corruption=0.5,
                batchswap_corruption=0.5,
            )
            ae.setup("fit")
            encoder = nn.ModuleList(
                [
                    BatchSwapNoise(0.5),
                    ResetSeed(seed),
                    Dropout(0.5),
                    nn.Linear(6, 3),
                    nn.ReLU(inplace=True),
                    nn.Linear(3, 2),
                    nn.ReLU(inplace=True),
                ]
            )
            decoder = nn.ModuleList(
                [
                    nn.Linear(2, 3),
                    nn.ReLU(inplace=True),
                    nn.Linear(3, 6),
                ]
            )
            self.assertEqual(ae.encoder.__repr__(), encoder.__repr__())
            self.assertEqual(ae.decoder.__repr__(), decoder.__repr__())

    def test_ColumnThreshold(self):
        data = tensor([[-1, -1], [0, 0], [1, 1]])
        with self.subTest("Empty indices"):
            steps = BinColumnThreshold(tensor([]).long())
            self.assertTrue(steps(data).equal(data))  # Do nothing

        with self.subTest("Set indices"):
            steps = BinColumnThreshold(tensor([0]).long())
            # sigmoid when x<0 is < 0.5, when x=0 == 0.5, x>0 > 0.5 (should be 1)
            self.assertTrue(steps(data).allclose(tensor([[0, -1], [1, 0], [1, 1]])))

        with self.subTest("Longitudinal"):
            data = tensor(
                [
                    [
                        [-1, -1],  # T = 0
                        [0, 0],  # T = 1
                        [1, 1],  # T = 2
                    ],  # End pt 1
                    [[-2, -2], [1, 1], [2, 2]],  # end pt 2
                ]
            )
            correct = tensor(
                [
                    [
                        [0, -1],  # T = 0
                        [1, 0],  # T = 1
                        [1, 1],  # T = 2
                    ],  # End pt 1
                    [[0, -2], [1, 1], [1, 2]],  # end pt 2
                ]
            )
            steps = BinColumnThreshold(tensor([0]).long())
            # sigmoid when x<0 is < 0.5, when x=0 == 0.5, x>0 > 0.5 (should be 1)
            self.assertTrue(steps(data).allclose(correct))

    def test_SoftmaxOnehot(self):
        data = tensor([[0, 1, 3.4, 9], [1, 3, 3.4, 9], [2, 5, 3.4, 9]])
        with self.subTest("Empty indices"):
            steps = OnehotColumnThreshold(tensor([]).long())
            self.assertTrue(steps(data).equal(data))  # Do nothing

        with self.subTest("1 set of indices"):
            steps = OnehotColumnThreshold(tensor([[0, 1]]).long())
            correct = tensor([[0, 1, 3.4, 9], [0, 1, 3.4, 9], [0, 1, 3.4, 9]])
            self.assertTrue(steps(data).allclose(correct))

        with self.subTest("Multiple Onehot Groups"):
            # the layer actually modifies the tensor in place so I ahve to do this again.
            data = tensor([[0, 1, 3.4, 9], [1, 3, 3.4, 9], [2, 5, 3.4, 9]])
            steps = OnehotColumnThreshold(tensor([[0, 1], [2, 3]]).long())
            correct = tensor([[0, 1, 0, 1], [0, 1, 0, 1], [0, 1, 0, 1]]).float()
            self.assertTrue(steps(data).allclose(correct))

        with self.subTest("Longitudinal"):
            data = tensor(
                [
                    [
                        [0, 1],  # T = 0
                        [1, 3],  # T = 1
                        [2, 5],  # T = 2
                    ],  # End pt 1
                    [[3.4, 9], [3.4, 9], [3.4, 9]],  # end pt 2
                ]
            )
            correct = tensor(
                [
                    [
                        [0, 1],  # T = 0
                        [0, 1],  # T = 1
                        [0, 1],  # T = 2
                    ],  # End pt 1
                    [[0, 1], [0, 1], [0, 1]],  # end pt 2
                ]
            ).float()
            steps = OnehotColumnThreshold(tensor([[0, 1]]).long())
            self.assertTrue(steps(data).allclose(correct))

    def test_loss(self):
        var = Variable(randn(10, 10), requires_grad=True)
        with self.subTest("CEMSELoss"):
            loss = CtnCatLoss(
                tensor([0, 1, 2, 3]).long(),
                tensor([4, 5, 6]).long(),
                tensor([[7, 8, 9]]).long(),
            )
            res = loss(var, var)
            try:  # should get no errors
                res.backward()
            except Exception as e:
                self.fail("Failed to call backward: " + e)
        with self.subTest("ReconstructionKLDivergenceLoss"):
            loss = ReconstructionKLDivergenceLoss(MSELoss())
            mu = Variable(randn(10, 1), requires_grad=True)
            logvar = Variable(randn(10, 1), requires_grad=True)
            res = loss(var, var, mu, logvar)
            try:  # should get no errors
                res.backward()
            except Exception as e:
                self.fail("Failed to call backward: " + e)


if __name__ == "__main__":
    unittest.main()

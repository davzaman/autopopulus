from argparse import Namespace
from typing import Dict, Optional, Union
import unittest
from unittest.mock import patch
import numpy as np
import torch

from torch.autograd import Variable
import torch.nn as nn
from torch import Generator, nan_to_num, rand, randn, tensor, isnan, long as torch_long
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.loss import BCEWithLogitsLoss, MSELoss
from torch.testing import assert_allclose
from pytorch_lightning.loggers.base import DummyLogger

import pandas as pd
from numpy.testing import assert_array_equal

from autopopulus.models.ap import AEImputer
from autopopulus.models.ae import AEDitto
from autopopulus.models.dnn import ResetSeed
from autopopulus.models.utils import (
    BatchSwapNoise,
    BinColumnThreshold,
    CtnCatLoss,
    OnehotColumnThreshold,
    ReconstructionKLDivergenceLoss,
)
from autopopulus.test.common_mock_data import (
    X,
    y,
    columns,
    col_idxs_by_type,
    groupby,
    seed,
)
from autopopulus.test.utils import get_dataset_loader
from autopopulus.data import CommonDataModule
from autopopulus.data.constants import PAD_VALUE
from autopopulus.data.transforms import list_to_tensor
from autopopulus.utils.log_utils import get_serialized_model_path

seed = 0
standard = {
    "hidden_layers": [3, 2, 3],
    "learning_rate": 0.1,
    "seed": seed,
}
data_settings = {
    "dataset_loader": get_dataset_loader(X["X"], y),
    "seed": seed,
    "val_test_size": 0.5,
    "test_size": 0.5,
    "batch_size": 2,
}
layer_dims = [6, 3, 2, 3, 6]
EPSILON = 1e-10


def get_data_args(
    col_idxs_set_empty: Optional[Dict] = None,
    groupby_set_empty: Optional[Dict] = None,
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
        "groupby": groupby,
        "columns": {"original": columns["columns"]},
        "discretizations": None,
        "inverse_target_encode_map": None,
    }
    if col_idxs_set_empty is not None:
        for key in col_idxs_set_empty:
            res["col_idxs_by_type"]["original"][key] = []

    if groupby_set_empty is not None:
        for key in groupby_set_empty:
            res["groupby"]["original"][key] = {}
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
    for high_level_metrics in self.metrics.values():
        for metric in high_level_metrics.values():
            assert metric.device == self.device
    return self.shared_step(batch, "train")[0]


class TestAEImputer(unittest.TestCase):
    @patch("autopopulus.data.dataset_classes.train_test_split")
    def test_basic(self, mock_split):
        mock_split.return_value = (X["X"].index, X["X"].index)
        datamodule = CommonDataModule(**data_settings, scale=True)
        aeimp = AEImputer(
            **standard, replace_nan_with=0, max_epochs=3, num_gpus=0, method="whatever"
        )
        # Turn off logging for testing
        aeimp.trainer.loggers = [DummyLogger()] if aeimp.trainer.loggers else []
        aeimp.fit(datamodule)
        train_dataloader = datamodule.train_dataloader()
        with self.subTest("Transform"):
            df = aeimp.transform(train_dataloader)

            self.assertEqual(df.isna().sum().sum(), 0)
            # the dataset is transformed so the values shouldn't be the same but the shape, columns, index should
            assert_array_equal(df.index, X["X"].index)
            assert_array_equal(df.columns, X["X"].columns)
            self.assertEqual(df.shape, X["X"].shape)
        with self.subTest("Load checkpoint"):
            other_aeimp = AEImputer.from_checkpoint(
                Namespace(
                    **standard,
                    replace_nan_with=0,
                    max_epochs=3,
                    num_gpus=0,
                    method="whatever",
                ),
                get_serialized_model_path(f"AEDitto_STATIC", "pt"),
            )
            other_df = other_aeimp.transform(train_dataloader)
            pd.testing.assert_frame_equal(other_df, df)
            self.assertEqual(other_aeimp.ae.hparams.keys(), aeimp.ae.hparams.keys())
            for k in other_aeimp.ae.hparams:
                item = other_aeimp.ae.hparams[k]
                if isinstance(item, dict):
                    if isinstance(list(item.values())[0], pd.Index):
                        self.assertEqual(item.keys(), aeimp.ae.hparams[k].keys())
                        for inner_k in item:
                            pd.testing.assert_index_equal(
                                item[inner_k], aeimp.ae.hparams[k][inner_k]
                            )
                    else:
                        np.testing.assert_equal(item, aeimp.ae.hparams[k])
                elif isinstance(item, list):
                    np.testing.assert_equal(item, aeimp.ae.hparams[k])
                else:
                    np.testing.assert_equal(item, aeimp.ae.hparams[k])
            self.assertDictEqual(
                other_aeimp.ae.hidden_and_cell_state, aeimp.ae.hidden_and_cell_state
            )
            self.assertEqual(
                str(other_aeimp.ae.state_dict()), str(aeimp.ae.state_dict())
            )
            # TODO: remove serialized stuff

    @torch.no_grad()
    @unittest.skipUnless(torch.cuda.is_available(), "No GPU was detected")
    @patch("autopopulus.data.dataset_classes.train_test_split")
    def test_device(self, mock_split):
        mock_split.return_value = (X["X"].index, X["X"].index)
        datamodule = CommonDataModule(**data_settings, scale=True)

        aeimp = AEImputer(
            **standard,
            replace_nan_with=0,
            max_epochs=3,
            num_gpus=1,  # check data, model, and metrics go on the gpu
        )
        # Turn off logging for testing
        aeimp.trainer.loggers = [DummyLogger()] if aeimp.trainer.loggers else []
        # TODO how do i ensure it's on the correct device with a unittest?
        # I need to test for the feature map inversions as  well
        with patch.object(AEDitto, "training_step", mock_training_step):
            aeimp.fit(datamodule)
        # just run it to see if anything i missed
        aeimp.fit(datamodule)


class TestAEDitto(unittest.TestCase):
    @patch("autopopulus.models.ae.onehot_column_threshold")
    @patch("autopopulus.models.ae.binary_column_threshold")
    def test_get_imputed_tensor_from_model_output(
        self,
        mock_binary_column_threshold,
        mock_onehot_column_threshold,
    ):
        ae = AEDitto(
            **standard,
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
            **standard,
            **get_data_args(
                col_idxs_set_empty=["categorical", "binary_vars", "onehot"],
                groupby_set_empty=["categorical_onehots", "binary"],
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
            **standard,
            **get_data_args(
                col_idxs_set_empty=["continuous", "categorical", "binary", "onehot"],
                groupby_set_empty=["categorical_onehots", "binary_vars"],
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
            new_settings = standard.copy()
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
                **standard,
                **get_data_args(),
                lossn="BCE",
            )
            ae.setup("fit")
            self.assertIsInstance(ae.loss, BCEWithLogitsLoss)

        # Dropout
        with self.subTest("Dropout"):
            ae = AEDitto(
                **standard,
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
            **standard,
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
                **standard,
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
                **standard,
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
                **standard,
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

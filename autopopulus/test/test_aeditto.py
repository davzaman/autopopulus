import unittest
from unittest.mock import patch

import torch.nn as nn
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.loss import BCEWithLogitsLoss

from pandas import Series

from autopopulus.models.ae import AEDitto
from autopopulus.models.dnn import ResetSeed
from autopopulus.models.utils import BatchSwapNoise, ReconstructionKLDivergenceLoss
from test.common_mock_data import indices, splits

seed = 0
standard = {
    "hidden_layers": [3, 2, 3],
    "learning_rate": 0.1,
    "seed": seed,
}
layer_dims = [7, 3, 2, 3, 7]
EPSILON = 1e-10


class TestAEDitto(unittest.TestCase):
    @patch("autopopulus.data.CommonDataModule")
    def mock_set_args_from_data(self, MockCommonDataModule):
        self.input_dim = 7
        if not hasattr(self, "ctn_cols_idx"):
            self.ctn_cols_idx = None
            self.cat_cols_idx = None
        MockCommonDataModule.return_value.splits = {
            "data": {"train": Series(splits["train"])}
        }
        self.datamodule = MockCommonDataModule()

    @patch.object(AEDitto, "set_args_from_data", mock_set_args_from_data)
    def test_ae(self):
        with self.assertRaises(AssertionError):
            AEDitto(**standard, vae=True, undiscretize_data=True).setup("fit")
        with self.assertRaises(AssertionError):
            AEDitto(**standard, lossn="BCEMSE", undiscretize_data=True).setup("fit")
        with self.assertRaises(AssertionError):  # need col idxs
            AEDitto(**standard, lossn="BCEMSE").setup("fit")

    @patch.object(AEDitto, "set_args_from_data", mock_set_args_from_data)
    def test_vae(self):
        ae = AEDitto(**standard, vae=True)
        ae.ctn_cols_idx = indices["ctn_cols"]
        ae.cat_cols_idx = indices["cat_cols"]
        ae.setup("fit")
        self.assertTrue(hasattr(ae, "fc_mu"))
        self.assertTrue(hasattr(ae, "fc_var"))
        self.assertIsInstance(ae.loss, ReconstructionKLDivergenceLoss)

        encoder = nn.ModuleList(
            [
                nn.Linear(7, 3),
                nn.ReLU(inplace=True),
            ]
        )
        mu_var = nn.Linear(3, 2)
        decoder = nn.ModuleList(
            [nn.Linear(2, 3), nn.ReLU(inplace=True), nn.Linear(3, 7)]
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
            2 * ((7 * 3) + (3 * 2)) + (3 * 2) + sum(layer_dims[1:]) + 2,
        )

    @patch.object(AEDitto, "set_args_from_data", mock_set_args_from_data)
    def test_basic(self):
        ae = AEDitto(**standard, lossn="BCE")
        ae.setup("fit")
        with self.subTest("Basic"):
            self.assertIsNone(ae.ctn_cols_idx)
            self.assertIsNone(ae.cat_cols_idx)

            self.assertFalse(hasattr(ae, "fc_mu"))
            self.assertFalse(hasattr(ae, "fc_var"))

            self.assertEqual(ae.code_index, 2)
            self.assertEqual(ae.layer_dims, layer_dims)

            encoder = nn.ModuleList(
                [
                    nn.Linear(7, 3),
                    nn.ReLU(inplace=True),
                    nn.Linear(3, 2),
                    nn.ReLU(inplace=True),
                ]
            )
            decoder = nn.ModuleList(
                [nn.Linear(2, 3), nn.ReLU(inplace=True), nn.Linear(3, 7)]
            )
            self.assertEqual(ae.encoder.__repr__(), encoder.__repr__())
            self.assertEqual(ae.decoder.__repr__(), decoder.__repr__())

            pytorch_total_params = sum(
                p.numel() for p in ae.parameters() if p.requires_grad
            )
            self.assertEqual(
                pytorch_total_params, 2 * ((7 * 3) + (3 * 2)) + sum(layer_dims[1:])
            )

        # Fractional hidden layer
        with self.subTest("Fractional hidden layer"):
            new_settings = standard.copy()
            new_settings["hidden_layers"] = [0.5, 0.1, 0.05, 0.1, 0.5]
            ae = AEDitto(**new_settings)
            ae.ctn_cols_idx = indices["ctn_cols"]
            ae.cat_cols_idx = indices["cat_cols"]
            ae.setup("fit")
            encoder = nn.ModuleList(
                [
                    nn.Linear(7, 4),
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
                    nn.Linear(4, 7),
                ]
            )
            self.assertEqual(ae.encoder.__repr__(), encoder.__repr__())
            self.assertEqual(ae.decoder.__repr__(), decoder.__repr__())

        # Loss Test
        with self.subTest("Loss"):
            ae = AEDitto(**standard, lossn="BCE")
            ae.ctn_cols_idx = indices["ctn_cols"]
            ae.cat_cols_idx = indices["cat_cols"]
            ae.setup("fit")
            self.assertIsInstance(ae.loss, BCEWithLogitsLoss)

        # Dropout
        with self.subTest("Dropout"):
            ae = AEDitto(**standard, dropout=0.5)
            ae.ctn_cols_idx = indices["ctn_cols"]
            ae.cat_cols_idx = indices["cat_cols"]
            ae.setup("fit")
            encoder = nn.ModuleList(
                [
                    nn.Linear(7, 3),
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
                    nn.Linear(3, 7),
                ]
            )
            self.assertEqual(ae.encoder.__repr__(), encoder.__repr__())
            self.assertEqual(ae.decoder.__repr__(), decoder.__repr__())

    @patch.object(AEDitto, "set_args_from_data", mock_set_args_from_data)
    def test_dae(self):
        ae = AEDitto(**standard, dropout_corruption=0.5)
        ae.ctn_cols_idx = indices["ctn_cols"]
        ae.cat_cols_idx = indices["cat_cols"]
        ae.setup("fit")
        encoder = nn.ModuleList(
            [
                ResetSeed(seed),
                Dropout(0.5),
                nn.Linear(7, 3),
                nn.ReLU(inplace=True),
                nn.Linear(3, 2),
                nn.ReLU(inplace=True),
            ]
        )
        decoder = nn.ModuleList(
            [
                nn.Linear(2, 3),
                nn.ReLU(inplace=True),
                nn.Linear(3, 7),
            ]
        )
        self.assertEqual(ae.encoder.__repr__(), encoder.__repr__())
        self.assertEqual(ae.decoder.__repr__(), decoder.__repr__())

        ae = AEDitto(**standard, batchswap_corruption=0.5)
        ae.ctn_cols_idx = indices["ctn_cols"]
        ae.cat_cols_idx = indices["cat_cols"]
        ae.setup("fit")
        encoder = nn.ModuleList(
            [
                BatchSwapNoise(0.5),
                nn.Linear(7, 3),
                nn.ReLU(inplace=True),
                nn.Linear(3, 2),
                nn.ReLU(inplace=True),
            ]
        )
        decoder = nn.ModuleList(
            [
                nn.Linear(2, 3),
                nn.ReLU(inplace=True),
                nn.Linear(3, 7),
            ]
        )
        self.assertEqual(ae.encoder.__repr__(), encoder.__repr__())
        self.assertEqual(ae.decoder.__repr__(), decoder.__repr__())

        ae = AEDitto(**standard, dropout_corruption=0.5, batchswap_corruption=0.5)
        ae.ctn_cols_idx = indices["ctn_cols"]
        ae.cat_cols_idx = indices["cat_cols"]
        ae.setup("fit")
        encoder = nn.ModuleList(
            [
                BatchSwapNoise(0.5),
                ResetSeed(seed),
                Dropout(0.5),
                nn.Linear(7, 3),
                nn.ReLU(inplace=True),
                nn.Linear(3, 2),
                nn.ReLU(inplace=True),
            ]
        )
        decoder = nn.ModuleList(
            [
                nn.Linear(2, 3),
                nn.ReLU(inplace=True),
                nn.Linear(3, 7),
            ]
        )
        self.assertEqual(ae.encoder.__repr__(), encoder.__repr__())
        self.assertEqual(ae.decoder.__repr__(), decoder.__repr__())


if __name__ == "__main__":
    unittest.main()

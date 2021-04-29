import unittest

import os
import sys

import torch.nn as nn
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.loss import BCEWithLogitsLoss

# For running tests in VSCode
sys.path.insert(
    1, os.path.join(sys.path[0], "path/to/autopopulus")
)
from models.ae import AEDitto
from models.dnn import ResetSeed
from models.utils import BatchSwapNoise, ReconstructionKLDivergenceLoss

seed = 0
standard = {
    "input_dim": 7,
    "hidden_layers": [3, 2, 3],
    "lr": 0.1,
    "seed": seed,
}
layer_dims = [7, 3, 2, 3, 7]
EPSILON = 1e-10
columns = ["age", "weight", "ismale", "fries_s", "fries_m", "fries_l"]


class TestAEDitto(unittest.TestCase):
    def setUp(self) -> None:
        return super().setUp()

    def test_ae(self):
        with self.assertRaises(AssertionError):
            AEDitto(**standard, vae=True, undiscretize_data=True)
        with self.assertRaises(AssertionError):
            AEDitto(**standard, lossn="BCEMSE", undiscretize_data=True)

    def test_vae(self):
        ae = AEDitto(**standard, vae=True)
        self.assertTrue(hasattr(ae, "fc_mu"))
        self.assertTrue(hasattr(ae, "fc_var"))
        self.assertIsInstance(ae.loss, ReconstructionKLDivergenceLoss)

        encoder = nn.Sequential(
            nn.Linear(7, 3),
            nn.ReLU(inplace=True),
        )
        mu_var = nn.Linear(3, 2)
        decoder = nn.Sequential(nn.Linear(2, 3), nn.ReLU(inplace=True), nn.Linear(3, 7))
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

    def test_basic(self):
        ae = AEDitto(**standard)
        with self.subTest("Basic"):
            self.assertIsNone(ae.columns)
            self.assertIsNone(ae.ctn_columns)
            self.assertIsNone(ae.ctn_cols_idx)
            self.assertIsNone(ae.cat_cols_idx)

            self.assertFalse(hasattr(ae, "fc_mu"))
            self.assertFalse(hasattr(ae, "fc_var"))

            self.assertEqual(ae.code_index, 2)
            self.assertEqual(ae.layer_dims, layer_dims)

            encoder = nn.Sequential(
                nn.Linear(7, 3),
                nn.ReLU(inplace=True),
                nn.Linear(3, 2),
                nn.ReLU(inplace=True),
            )
            decoder = nn.Sequential(
                nn.Linear(2, 3), nn.ReLU(inplace=True), nn.Linear(3, 7)
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
            with self.assertRaises(Exception):
                # will intepret as int and then do integer truncation on 0.5 (resulting in layer size 0)
                standard["hidden_layers"] = [1, 0.5, 1]
                AEDitto(**standard)

            standard["hidden_layers"] = [0.5, 0.1, 0.05, 0.1, 0.5]
            ae = AEDitto(**standard)
            encoder = nn.Sequential(
                nn.Linear(7, 4),
                nn.ReLU(inplace=True),
                nn.Linear(4, 1),
                nn.ReLU(inplace=True),
                nn.Linear(1, 1),
                nn.ReLU(inplace=True),
            )
            decoder = nn.Sequential(
                nn.Linear(1, 1),
                nn.ReLU(inplace=True),
                nn.Linear(1, 4),
                nn.ReLU(inplace=True),
                nn.Linear(4, 7),
            )
            self.assertEqual(ae.encoder.__repr__(), encoder.__repr__())
            self.assertEqual(ae.decoder.__repr__(), decoder.__repr__())

        # Loss Test
        with self.subTest("Loss"):
            ae = AEDitto(**standard, lossn="BCE")
            self.assertIsInstance(ae.loss, BCEWithLogitsLoss)

        # Dropout
        with self.subTest("Dropout"):
            ae = AEDitto(**standard, dropout=0.5)
            encoder = nn.Sequential(
                nn.Linear(7, 3),
                nn.ReLU(inplace=True),
                ResetSeed(seed),
                Dropout(0.5),
                nn.Linear(3, 2),
                nn.ReLU(inplace=True),
                ResetSeed(seed),
                Dropout(0.5),
            )
            decoder = nn.Sequential(
                nn.Linear(2, 3),
                nn.ReLU(inplace=True),
                ResetSeed(seed),
                Dropout(0.5),
                nn.Linear(3, 7),
            )
            self.assertEqual(ae.encoder.__repr__(), encoder.__repr__())
            self.assertEqual(ae.decoder.__repr__(), decoder.__repr__())

    def test_dae(self):
        ae = AEDitto(**standard, dropout_corruption=0.5)
        encoder = nn.Sequential(
            ResetSeed(seed),
            Dropout(0.5),
            nn.Linear(7, 3),
            nn.ReLU(inplace=True),
            nn.Linear(3, 2),
            nn.ReLU(inplace=True),
        )
        decoder = nn.Sequential(
            nn.Linear(2, 3),
            nn.ReLU(inplace=True),
            nn.Linear(3, 7),
        )
        self.assertEqual(ae.encoder.__repr__(), encoder.__repr__())
        self.assertEqual(ae.decoder.__repr__(), decoder.__repr__())

        ae = AEDitto(**standard, batchswap_corruption=0.5)
        encoder = nn.Sequential(
            BatchSwapNoise(0.5),
            nn.Linear(7, 3),
            nn.ReLU(inplace=True),
            nn.Linear(3, 2),
            nn.ReLU(inplace=True),
        )
        decoder = nn.Sequential(
            nn.Linear(2, 3),
            nn.ReLU(inplace=True),
            nn.Linear(3, 7),
        )
        self.assertEqual(ae.encoder.__repr__(), encoder.__repr__())
        self.assertEqual(ae.decoder.__repr__(), decoder.__repr__())

        ae = AEDitto(**standard, dropout_corruption=0.5, batchswap_corruption=0.5)
        encoder = nn.Sequential(
            BatchSwapNoise(0.5),
            ResetSeed(seed),
            Dropout(0.5),
            nn.Linear(7, 3),
            nn.ReLU(inplace=True),
            nn.Linear(3, 2),
            nn.ReLU(inplace=True),
        )
        decoder = nn.Sequential(
            nn.Linear(2, 3),
            nn.ReLU(inplace=True),
            nn.Linear(3, 7),
        )
        self.assertEqual(ae.encoder.__repr__(), encoder.__repr__())
        self.assertEqual(ae.decoder.__repr__(), decoder.__repr__())


if __name__ == "__main__":
    unittest.main()

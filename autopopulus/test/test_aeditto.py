import unittest
from unittest.mock import patch

from torch.autograd import Variable
import torch.nn as nn
from torch import rand, randn, Generator, Tensor
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.loss import BCEWithLogitsLoss, MSELoss

from pandas import Series

from autopopulus.models.ae import AEDitto
from autopopulus.models.dnn import ResetSeed
from autopopulus.models.utils import (
    BatchSwapNoise,
    BinColumnThreshold,
    CtnCatLoss,
    OnehotColumnThreshold,
    ReconstructionKLDivergenceLoss,
)
from autopopulus.test.common_mock_data import splits

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
        self.nfeatures = {"original": 7}
        self.col_idxs_by_type = {
            "original": {
                "continuous": [0, 5],
                "categorical": [1, 2, 3, 4, 6],
                "binary": [1, 6],
                "onehot": [[2, 3, 4]],
            }
        }
        if hasattr(self, "col_idxs_set_empty"):
            for key in self.col_idxs_set_empty:
                self.col_idxs_by_type["original"][key] = []

        self.groupby = {
            "original": {
                "categorical_onehots": {2: "A", 3: "A", 4: "A"},
                "binary_vars": {1: "b1", 6: "b2"},
            }
        }
        if hasattr(self, "groupby_set_empty"):
            for key in self.groupby_set_empty:
                self.groupby["original"][key] = {}

        MockCommonDataModule.return_value.splits = {
            "data": {"train": Series(splits["train"])}
        }
        self.datamodule = MockCommonDataModule()

    @patch.object(AEDitto, "set_args_from_data", mock_set_args_from_data)
    def test_vae(self):
        ae = AEDitto(**standard, variational=True, lossn="MSE")
        ae.col_idxs_set_empty = ["categorical", "binary_vars", "onehot"]
        ae.groupby_set_empty = ["categorical_onehots", "binary"]
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
        ae.col_idxs_set_empty = ["continuous", "categorical", "binary", "onehot"]
        ae.groupby_set_empty = ["categorical_onehots", "binary_vars"]
        ae.setup("fit")
        with self.subTest("Basic"):
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
                [
                    nn.Linear(2, 3),
                    nn.ReLU(inplace=True),
                    nn.Linear(3, 7),
                ]
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
            ae.setup("fit")
            self.assertIsInstance(ae.loss, BCEWithLogitsLoss)

        # Dropout
        with self.subTest("Dropout"):
            ae = AEDitto(**standard, dropout=0.5)
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
    def test_longitudinal(self):
        ae = AEDitto(**standard, longitudinal=True)
        ae.setup("fit")
        encoder = nn.ModuleList(
            [
                nn.LSTM(7, 3, batch_first=True),
                nn.ReLU(inplace=True),
                nn.LSTM(3, 2, batch_first=True),
                nn.ReLU(inplace=True),
            ]
        )
        decoder = nn.ModuleList(
            [
                nn.LSTM(2, 3, batch_first=True),
                nn.ReLU(inplace=True),
                nn.Linear(3, 7),
            ]
        )
        self.assertEqual(ae.encoder.__repr__(), encoder.__repr__())
        self.assertEqual(ae.decoder.__repr__(), decoder.__repr__())

        with self.subTest("Apply Layers"):
            X_long = rand(size=(7, 7, 7), generator=Generator().manual_seed(seed))
            code = ae.encode("train", X_long, Tensor([7] * 7))
            self.assertEqual(ae.curr_rnn_depth, 1)

            ae.decode("train", code, Tensor([7] * 7))
            self.assertEqual(ae.curr_rnn_depth, 0)

    @patch.object(AEDitto, "set_args_from_data", mock_set_args_from_data)
    def test_dae(self):
        with self.subTest("Dropout Corruption"):
            ae = AEDitto(**standard, dropout_corruption=0.5)
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

        with self.subTest("Batchswap Corruption"):
            ae = AEDitto(**standard, batchswap_corruption=0.5)
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

        with self.subTest("Dropout and Batchswap Corruption"):
            ae = AEDitto(**standard, dropout_corruption=0.5, batchswap_corruption=0.5)
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

    def test_ColumnThreshold(self):
        data = Tensor([[-1, -1], [0, 0], [1, 1]])
        with self.subTest("Empty indices"):
            steps = BinColumnThreshold(Tensor([]).long())
            self.assertTrue(steps(data).equal(data))  # Do nothing

        with self.subTest("Set indices"):
            steps = BinColumnThreshold(Tensor([0]).long())
            # sigmoid when x<0 is < 0.5, when x=0 == 0.5, x>0 > 0.5 (should be 1)
            self.assertTrue(steps(data).allclose(Tensor([[0, -1], [1, 0], [1, 1]])))

        with self.subTest("Longitudinal"):
            data = Tensor(
                [
                    [
                        [-1, -1],  # T = 0
                        [0, 0],  # T = 1
                        [1, 1],  # T = 2
                    ],  # End pt 1
                    [[-2, -2], [1, 1], [2, 2]],  # end pt 2
                ]
            )
            correct = Tensor(
                [
                    [
                        [0, -1],  # T = 0
                        [1, 0],  # T = 1
                        [1, 1],  # T = 2
                    ],  # End pt 1
                    [[0, -2], [1, 1], [1, 2]],  # end pt 2
                ]
            )
            steps = BinColumnThreshold(Tensor([0]).long())
            # sigmoid when x<0 is < 0.5, when x=0 == 0.5, x>0 > 0.5 (should be 1)
            self.assertTrue(steps(data).allclose(correct))

    def test_SoftmaxOnehot(self):
        data = Tensor([[0, 1, 3.4, 9], [1, 3, 3.4, 9], [2, 5, 3.4, 9]])
        with self.subTest("Empty indices"):
            steps = OnehotColumnThreshold(Tensor([]).long())
            self.assertTrue(steps(data).equal(data))  # Do nothing

        with self.subTest("1 set of indices"):
            steps = OnehotColumnThreshold(Tensor([[0, 1]]).long())
            correct = Tensor([[0, 1, 3.4, 9], [0, 1, 3.4, 9], [0, 1, 3.4, 9]])
            self.assertTrue(steps(data).allclose(correct))

        with self.subTest("Multiple Onehot Groups"):
            # the layer actually modifies the tensor in place so I ahve to do this again.
            data = Tensor([[0, 1, 3.4, 9], [1, 3, 3.4, 9], [2, 5, 3.4, 9]])
            steps = OnehotColumnThreshold(Tensor([[0, 1], [2, 3]]).long())
            correct = Tensor([[0, 1, 0, 1], [0, 1, 0, 1], [0, 1, 0, 1]])
            self.assertTrue(steps(data).allclose(correct))

        with self.subTest("Longitudinal"):
            data = Tensor(
                [
                    [
                        [0, 1],  # T = 0
                        [1, 3],  # T = 1
                        [2, 5],  # T = 2
                    ],  # End pt 1
                    [[3.4, 9], [3.4, 9], [3.4, 9]],  # end pt 2
                ]
            )
            correct = Tensor(
                [
                    [
                        [0, 1],  # T = 0
                        [0, 1],  # T = 1
                        [0, 1],  # T = 2
                    ],  # End pt 1
                    [[0, 1], [0, 1], [0, 1]],  # end pt 2
                ]
            )
            steps = OnehotColumnThreshold(Tensor([[0, 1]]).long())
            self.assertTrue(steps(data).allclose(correct))

    def test_loss(self):
        var = Variable(randn(10, 10), requires_grad=True)
        with self.subTest("CEMSELoss"):
            loss = CtnCatLoss(
                Tensor([0, 1, 2, 3]).long(),
                Tensor([4, 5, 6]).long(),
                Tensor([[7, 8, 9]]).long(),
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

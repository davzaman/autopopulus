import unittest
import numpy as np
import pandas as pd
import torch

from autopopulus.utils.impute_metrics import AccuracyPerBin, MAAPE, RMSE, EPSILON
from autopopulus.utils.utils import div0, flatten_groupby
from test.common_mock_data import columns, X, seed, groupby, discretization

WITHIN = 6
standard = {
    "input_dim": 7,
    "hidden_layers": [3, 2, 3],
    "lr": 0.1,
    "seed": seed,
}
layer_dims = [7, 3, 2, 3, 7]


X_est = pd.DataFrame(
    # This does not match X["nomissing"]
    [
        [44, 14.3, 0, 0, 1, 0],
        [49, 57.2, 1, 0, 0, 1],
        [26, 26.3, 0, 0, 0, 1],
        [16, 73.4, 1, 1, 0, 0],
        [22, 54.9, 1, 0, 1, 0],
        [57, 29.6, 0, 1, 0, 0],
    ],
    columns=columns["columns"],
)
X_est2 = pd.DataFrame(
    [
        [-500, 14.3, 0, 0, 1, 0],
        [-500, 57.2, 1, 0, 0, 1],
        [-500, 26.3, 0, 0, 0, 1],
        [-500, 73.4, 1, 1, 0, 0],
        [22, 54.9, 1, 0, 1, 0],
        [-500, 29.6, 0, 1, 0, 0],
    ],
    columns=columns["columns"],
)


nelements_missing = X["X"].isna().sum(axis=1)
summed_squared_errors = np.array(
    [
        ((15.1 - 14.3) ** 2),
        0,
        ((0 - 0) ** 2 + (1 - 0) ** 2 + (0 - 1) ** 2),
        0,
        ((13 - 22) ** 2 + (56.5 - 54.9) ** 2),
        0,
    ]
)
mask_summed_squared_errors = summed_squared_errors.copy()
# ignore the one that's not in the missing mask
mask_summed_squared_errors[-2] -= (56.5 - 54.9) ** 2

summed_maape_precomp = np.array(
    [
        np.arctan(np.abs((15.1 - 14.3) / (15.1 + EPSILON))),
        0,
        np.arctan(np.abs((0 - 0) / (0 + EPSILON)))
        + np.arctan(np.abs((1 - 0) / (1 + EPSILON)))
        + np.arctan(np.abs((0 - 1) / (0 + EPSILON))),
        0,
        np.arctan(np.abs((13 - 22) / (13 + EPSILON)))
        + np.arctan(np.abs((56.5 - 54.9) / (56.5 + EPSILON))),
        0,
    ]
)
mask_summed_maape_precomp = summed_maape_precomp.copy()
# ignore the one that's not in the missing mask
mask_summed_maape_precomp[-2] -= np.arctan(np.abs((56.5 - 54.9) / (56.5 + EPSILON)))

rmse_true = np.sqrt(np.mean(summed_squared_errors / 6))
maape_true = np.mean(summed_maape_precomp / 6)

rmse_true_missing_only = np.sqrt(
    np.mean(div0(mask_summed_squared_errors, nelements_missing))
)
maape_true_missing_only = np.mean(div0(mask_summed_maape_precomp, nelements_missing))


class TestMetrics(unittest.TestCase):
    def setUp(self) -> None:
        return super().setUp()

    def test_np_metric(self):
        # the metrics should work for numpy too
        self.assertEqual(0, RMSE(X["nomissing"], X["nomissing"]).item())
        self.assertEqual(0, MAAPE(X["nomissing"], X["nomissing"]).item())
        self.assertEqual(rmse_true, RMSE(X_est, X["nomissing"]).item())
        self.assertAlmostEqual(maape_true, MAAPE(X_est, X["nomissing"]).item())
        missing_mask = X["X"].isna()
        self.assertEqual(0, RMSE(X["nomissing"], X["nomissing"], missing_mask).item())
        self.assertEqual(0, MAAPE(X["nomissing"], X["nomissing"], missing_mask).item())
        self.assertEqual(
            rmse_true_missing_only, RMSE(X_est, X["nomissing"], missing_mask).item()
        )
        self.assertEqual(
            maape_true_missing_only,
            MAAPE(X_est, X["nomissing"], missing_mask).item(),
        )
        self.assertEqual(
            rmse_true_missing_only, RMSE(X_est2, X["nomissing"], missing_mask).item()
        )
        self.assertEqual(
            maape_true_missing_only,
            MAAPE(X_est2, X["nomissing"], missing_mask).item(),
        )

    def test_tensor_metric(self):
        # when subtracting, tensors add in a little margin of error that accumulates, so we want to get close within WITHIN decimal places.
        WITHIN = 6
        X_nomissing_tensor = torch.tensor(X["nomissing"].values)
        X_est_tensor = torch.tensor(X_est.values)
        X_est_tensor2 = torch.tensor(X_est2.values)
        with self.subTest("All Data"):
            self.assertAlmostEqual(
                0,
                RMSE(X_nomissing_tensor, X_nomissing_tensor).item(),
                places=WITHIN,
            )
            self.assertAlmostEqual(
                0,
                MAAPE(X_nomissing_tensor, X_nomissing_tensor).item(),
                places=WITHIN,
            )

            # Now if they dont exactly equal each other
            with self.subTest("Not Equal"):
                self.assertAlmostEqual(
                    rmse_true,
                    RMSE(X_est_tensor, X_nomissing_tensor).item(),
                    places=WITHIN,
                )
                self.assertAlmostEqual(
                    maape_true,
                    MAAPE(X_est_tensor, X_nomissing_tensor).item(),
                    places=WITHIN,
                )

        with self.subTest("Originally Missing Only"):
            missing_mask = torch.tensor(X["X"].values).isnan()
            self.assertAlmostEqual(
                0,
                RMSE(X_nomissing_tensor, X_nomissing_tensor, missing_mask).item(),
                places=WITHIN,
            )
            self.assertAlmostEqual(
                0,
                MAAPE(X_nomissing_tensor, X_nomissing_tensor, mask=missing_mask).item(),
                places=WITHIN,
            )

            # Now if they dont exactly equal each other
            with self.subTest("Not Equal"):
                self.assertAlmostEqual(
                    rmse_true_missing_only,
                    RMSE(X_est_tensor, X_nomissing_tensor, missing_mask).item(),
                    places=WITHIN,
                )
                self.assertAlmostEqual(
                    maape_true_missing_only,
                    MAAPE(X_est_tensor, X_nomissing_tensor, mask=missing_mask).item(),
                    places=WITHIN,
                )

                # make sure that the value is still the same even if the values outside the mask don't match, since we don't care about them and don't want to count them
                with self.subTest("Not Equal outside mask"):
                    self.assertAlmostEqual(
                        rmse_true_missing_only,
                        RMSE(X_est_tensor2, X_nomissing_tensor, missing_mask).item(),
                        places=WITHIN,
                    )
                    self.assertAlmostEqual(
                        maape_true_missing_only,
                        MAAPE(
                            X_est_tensor2, X_nomissing_tensor, mask=missing_mask
                        ).item(),
                        places=WITHIN,
                    )

    def test_accuracy_per_bin(self):
        # TODO: add in numpy test?

        X_disc = X["disc"].copy()
        # make the second to last sample be missing the weights too
        X_disc.iloc[
            -2, discretization["discretizer_dict"]["weight"]["indices"]
        ] = np.nan
        X_disc = torch.tensor(X_disc.values)
        flattened_groupby = flatten_groupby(groupby["after_fit"]["discretize"])

        missing_mask = X_disc.isnan()
        # 4 vars, 6 features, 2 incorrect
        accuracy_per_bin = ((4 * 6) - 2) / (4 * 6)
        # 4 values for a feature missing, 2 correct
        accuracy_per_bin_missing_only = 2 / 4

        X_disc_est = torch.tensor(
            [
                [0, 0, 1, 0, 0, 0, 1, 0.7, 0.05, 0.05, 0.2],  # correct
                [1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1],
                [0, 0.7, 0.1, 0.1, 0, 1, 0, 0, 1, 0, 0],  # incorrect
                [1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1],
                [1, 0, 1, 0, 0.7, 0.1, 0.1, 0.6, 0.1, 0.1, 0.1],  # correct, incorrect
                [0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0],
            ],
        )
        X_disc_est_2 = torch.tensor(
            [
                [0, 0, 1, 0, 0, 0, 1, 0.7, 0.05, 0.05, 0.2],  # correct
                [1, 0, 0, 1, 0, 1, 0, 0.2, 0, 0, 1],
                [0, 0.7, 0.1, 0.1, 0, 1, 0, 0, 1, 0, 0],  # incorrect
                [1, 1, 0, 0, 1, 0, 0, 0, 0, 0.3, 1],
                [1, 0, 1, 0, 0.7, 0.1, 0.1, 0.6, 0.1, 0.1, 0.1],  # correct, incorrect
                [0.1, 0.9, 0, 0, 0, 0, 1, 0, 1, 0, 0],
            ],
        )

        self.assertAlmostEqual(
            1,
            AccuracyPerBin(
                torch.tensor(X["disc_true"].values),
                torch.tensor(X["disc_true"].values),
                flattened_groupby,
                missing_mask,
            ),
            places=WITHIN,
        )

        self.assertAlmostEqual(
            accuracy_per_bin,
            AccuracyPerBin(
                X_disc_est,
                torch.tensor(X["disc_true"].values),
                flattened_groupby,
            ),
            places=WITHIN,
        )

        self.assertAlmostEqual(
            1,
            AccuracyPerBin(
                torch.tensor(X["disc_true"].values),
                torch.tensor(X["disc_true"].values),
                flattened_groupby,
                missing_mask,
            ),
            places=WITHIN,
        )

        # Now if they dont exactly equal each other
        self.assertAlmostEqual(
            accuracy_per_bin_missing_only,
            AccuracyPerBin(
                X_disc_est,
                torch.tensor(X["disc_true"].values),
                flattened_groupby,
                missing_mask,
            ),
            places=WITHIN,
        )

        # make sure that the value is still the same even if the values outside the mask don't match, since we don't care about them and don't want to count them
        self.assertAlmostEqual(
            accuracy_per_bin_missing_only,
            AccuracyPerBin(
                X_disc_est_2,
                torch.tensor(X["disc_true"].values),
                flattened_groupby,
                missing_mask,
            ),
            places=WITHIN,
        )


if __name__ == "__main__":
    unittest.main()

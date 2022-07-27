import unittest
import pandas as pd
import numpy as np

from autopopulus.data.transforms import ampute
from test.common_mock_data import seed, X, columns


class TestAmpute(unittest.TestCase):
    def setUp(self) -> None:
        return super().setUp()

    def testAmputeGeneral(self):
        # not specifying a mechanism changes nothing
        self.assertTrue(
            X["nomissing"].equals(
                ampute(X["nomissing"], seed, missing_cols=columns["ctn_cols"], mech="")
            )
        )

        percent = 0.28
        missing_cols = [13, 8]
        observed_cols = [2, 9]

        # The DF has to be large enough or the random masking will not work well with percentage missing.
        rand = pd.DataFrame(np.random.random((300, 15)))
        mcar = ampute(
            rand,
            seed,
            missing_cols=missing_cols,
            percent=percent,
            mech="MCAR",
        )
        mar = ampute(
            rand,
            seed,
            missing_cols=missing_cols,
            percent=percent,
            mech="MAR",
            observed_cols=observed_cols,
        )
        mnar1 = ampute(
            rand,
            seed,
            missing_cols=missing_cols,
            percent=percent,
            mech="MNAR1",
        )
        mnar = ampute(
            rand,
            seed,
            missing_cols=missing_cols,
            percent=percent,
            mech="MNAR",
        )

        # make sure all mechanisms produce something different
        self.assertFalse(mcar.equals(mar))
        self.assertFalse(mcar.equals(mnar))
        self.assertFalse(mcar.equals(mnar1))
        self.assertFalse(mar.equals(mnar))
        self.assertFalse(mar.equals(mnar1))
        self.assertFalse(mnar.equals(mnar1))

        # check proper amount missing
        for amputed in [mcar, mar, mnar, mnar1]:
            missing_per_col = amputed[missing_cols].isna().sum() / amputed.shape[0]
            for amount_missing in missing_per_col:
                self.assertAlmostEqual(percent, amount_missing, delta=0.05)

    def testMCAR(self):
        # Fail to specify missing columns to give an error
        with self.assertRaises(TypeError):
            ampute(X["nomissing"], seed, mech="MCAR")

        percent = 0.28
        missing_cols = [13, 8]
        # The DF has to be large enough or the random masking will not work well with percentage missing.
        ones = pd.DataFrame(np.ones((300, 15)))
        amputed = ampute(
            ones, seed, missing_cols=missing_cols, percent=percent, mech="MCAR"
        )

        # original df not modified
        self.assertFalse(ones.equals(amputed))

        # should produce the same result with same settings (+ same seed)
        self.assertTrue(
            amputed.equals(
                # same settings as before
                ampute(
                    ones, seed, missing_cols=missing_cols, percent=percent, mech="MCAR"
                )
            )
        )

    def testMAR(self):
        # Fail to specify missing columns to give an error
        with self.assertRaises(TypeError):
            ampute(X["nomissing"], seed, mech="MAR")

        percent = 0.28
        missing_cols = [13, 8]
        observed_cols = [2, 9]

        # Fail to specify observed vals will give an error
        with self.assertRaises(TypeError):
            ampute(X["nomissing"], seed, missing_cols=missing_cols, mech="MAR")

        # The DF has to be large enough or the random masking will not work well with percentage missing.
        ones = pd.DataFrame(np.ones((300, 15)))
        rand = pd.DataFrame(np.random.random((300, 15)))
        amputed = ampute(
            rand,
            seed,
            missing_cols=missing_cols,
            percent=percent,
            mech="MAR",
            observed_cols=observed_cols,
        )
        amputed_ones = ampute(
            ones,
            seed,
            missing_cols=missing_cols,
            percent=percent,
            mech="MAR",
            observed_cols=observed_cols,
        )

        # original df not modified
        self.assertFalse(rand.equals(amputed))
        # if there's no dist (all 0s) nothing will happen
        self.assertTrue(ones.equals(amputed_ones))

        # should produce the same result with same settings (+ same seed)
        self.assertTrue(
            amputed.equals(
                # same settings as before
                ampute(
                    rand,
                    seed,
                    missing_cols=missing_cols,
                    percent=percent,
                    mech="MAR",
                    observed_cols=observed_cols,
                )
            )
        )

        # TODO: tests for MNAR + MNAR1


if __name__ == "__main__":
    unittest.main()

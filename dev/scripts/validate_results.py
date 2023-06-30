# %%
import numpy as np
import pandas as pd

from utils import (
    EXPERIMENT_GRID_VARS,
    IMPUTE_METRIC_DIMENSIONS,
    PREDICT_METRIC_DIMENSIONS,
    all_impute_but,
    all_predict_but,
)

tracker = "mlflow"

"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Impute Metric Validation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""
# %%
impute_data = pd.read_pickle(
    f"/home/davina/Private/repos/autopopulus/guild_runs/{tracker}_impute_results.pkl"
)

# %%
test_data = impute_data[(impute_data["split"] == "test")].set_index(
    EXPERIMENT_GRID_VARS
)[["val"] + IMPUTE_METRIC_DIMENSIONS]
# Evaluate loss separately
test_metrics = test_data[test_data["metric_name"] != "loss"]

############################
#      Global asserts      #
############################
# %%
#### Test filter_subgroup ####
assert all(
    test_metrics.groupby(all_impute_but("filter_subgroup"))[
        ["val", "filter_subgroup"]
    ].apply(
        lambda group: np.array_equal(
            group["filter_subgroup"].values, ["all", "missingonly"]
        )
    )
), "Each metric should have an 'all' version and 'missingonly' version"
assert all(
    test_metrics.groupby(all_impute_but("filter_subgroup"))[
        ["val", "filter_subgroup"]
    ].apply(
        lambda group: group[group["filter_subgroup"] == "all"]
        < group[group["filter_subgroup"] == "missingonly"]
    )
), "all < missingonly for everything"

# %%
assert not any(
    test_metrics[test_metrics["feature_type"] == "mixed"]["val"] == 0
), "We shouldn't have combined errors be 0 ever (it's possible, but would be weird if the imputation had 0 error; investigate.)"

# %%
assert all(
    test_metrics[test_metrics["feature_type"] == "continuous"]
    .groupby(all_impute_but("metric_name"))["val"]
    .apply(lambda x: (vals := set(x)) == {0} or len(vals) == 2)
), "RMSE != MAAPE for continuous portions of metrics, except for when there is no continuous component (discretize_continuous) in which case it'll be all 0."

# This doesn't have to be 0 because when amputing we're only making 1 column missing.
print(
    f"Percent of RMSE metrics where CW == EW: {( test_metrics[ (test_metrics['filter_subgroup'] == 'missingonly') & test_metrics['metric_name'].str.contains('RMSE') ] .groupby(all_impute_but('reduction'))[['val', 'reduction']] .apply(lambda x: x[x['reduction'] == 'CW'] == x[x['reduction'] == 'EW']))['val'].mean()*100}"
)

#############################
#      Feature Mapping      #
#############################
# %%
when_disc = test_metrics[
    test_metrics.index.get_level_values("feature-map") == "discretize_continuous"
]
assert all(
    when_disc[
        (when_disc["feature_space"] == "mapped")
        & (when_disc["feature_type"] == "continuous")
    ]
    == 0
), "When discretize_continuous the continuous metrics in mapped space should all be 0."

assert all(
    when_disc[when_disc["feature_space"] == "mapped"]
    .groupby(all_impute_but("feature_type"))[["val", "feature_type"]]
    .apply(
        lambda x: all(
            x[x["feature_type"] == "mixed"] == x[x["feature_type"] == "categorical"]
        )
    )
), "When discretize_continuous in mapped space the categorical metrics == mixed metrics (all other dims held equal)."

assert all(
    when_disc[
        (when_disc["feature_space"] == "mapped")
        & (when_disc["feature_type"] == "mixed")
    ]
    .groupby(EXPERIMENT_GRID_VARS + ["filter_subgroup"])["val"]
    .apply(lambda x: all(np.isclose(x, x[0])))
), "When discretize_continuous, the mixed metrics in mapped space should all be the same (or very close/float imprecision)."

# %%
when_targ_enc = test_metrics[
    test_metrics.index.get_level_values("feature-map") == "target_encode_categorical"
]
assert all(
    when_targ_enc[
        (when_targ_enc["feature_space"] == "mapped")
        & (when_targ_enc["feature_type"] == "categorical")
    ]
    == 0
), "When target_encode_categorical the categorical metrics in mapped space should all be 0."

assert all(
    when_targ_enc[when_targ_enc["feature_space"] == "mapped"]
    .groupby(all_impute_but("feature_type"))[["val", "feature_type"]]
    .apply(
        lambda x: all(
            x[x["feature_type"] == "mixed"] == x[x["feature_type"] == "continuous"]
        )
    )
), "When target_encode_categorical in mapped space the continuous metrics == mixed metrics (all other dims held equal)."

"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Predict Metric Validation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""
# %%
predict_data = pd.read_pickle(
    f"/home/davina/Private/repos/autopopulus/guild_runs/{tracker}_predict_results.pkl"
)
predict_data = predict_data.set_index(EXPERIMENT_GRID_VARS)[
    ["val"] + PREDICT_METRIC_DIMENSIONS
]
# %%
assert all(
    (
        nlogged := predict_data.groupby(
            EXPERIMENT_GRID_VARS + PREDICT_METRIC_DIMENSIONS
        )["val"].size()
    )
    == nlogged[0]
), "All metrics should be logged the same amount of times."
assert nlogged[0] > 1, "Predict should have been bootstrap sampled more than 1x."


# %%
assert all(
    (
        metrics_logged := predict_data.groupby(all_predict_but("metric_name"))[
            "metric_name"
        ].apply(lambda x: x.unique())
    ).apply(lambda x: np.array_equal(x, metrics_logged.values[0]))
), "All experiments should have the same metrics logged."

# %%
assert np.array_equal(
    metrics_logged.values[0],
    np.array(  # Align with evaluate() method in prediction_models.py
        [
            "F1-score",
            "Recall-score",
            "Precision-score",
            "ROC-AUC",
            "PR-AUC",
            "Brier-score",
            "TP",
            "TN",
            "FP",
            "FN",
        ]
    ),
), "Not logging all the metrics we want to for prediction."

# %%

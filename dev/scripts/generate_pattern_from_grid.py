import json

# DATASETS=[cure_ckd, crrt]
DATASETS = ["cure_ckd"]
MISSINGNESS_MECHANISMS = ["MCAR", "MAR", "MNAR(G)", "MNAR(Y)", "MNAR"]
SCORE_TO_PROBABILITY_FUNCTIONS = ["sigmoid-mid", "sigmoid-tail"]
# NUM_INCOMPLETE = ["1", "many"]
# NUM_OBSERVED = ["1", "many"]
NUM_INCOMPLETE = ["many"]
NUM_OBSERVED = ["many"]
FEATURES_INVOLVED = {
    "cure_ckd": {
        "static": {
            "missing": [
                "egfr_entry_period_hba1c_mean",
                "egfr_entry_htn_flag"
                # "egfr_entry_period_uacr_mean",
                # "egfr_entry_period_sbp_mean",
            ],
            "influence_missing": ["egfr_entry_age", "egfr_entry_period_av_count"],
        }
    }
}


def create_pattern(
    incomplete_vars,
    mechanism,
    observed_vars,
    score_to_probability_func,
    num_incomplete,
    num_observed,
):
    if num_incomplete == "1":
        incomplete_vars = [incomplete_vars[0]]
    pattern = {
        "incomplete_vars": incomplete_vars,
        "mechanism": mechanism,
        "score_to_probability_func": score_to_probability_func,
    }
    if mechanism == "MAR":
        if num_observed == "1":
            observed_vars = [observed_vars[0]]
        pattern["weights"] = {var: 1 for var in observed_vars}
    return pattern


if __name__ == "__main__":
    patterns = []
    for dataset in DATASETS:
        incomplete_vars = FEATURES_INVOLVED[dataset]["static"]["missing"]
        observed_vars = FEATURES_INVOLVED[dataset]["static"]["influence_missing"]
        for score_to_probability_func in SCORE_TO_PROBABILITY_FUNCTIONS:
            for mechanism in MISSINGNESS_MECHANISMS:
                for num_incomplete in NUM_INCOMPLETE:
                    if mechanism == "MAR":
                        for num_observed in NUM_OBSERVED:
                            pattern = create_pattern(
                                incomplete_vars,
                                mechanism,
                                observed_vars,
                                score_to_probability_func,
                                num_incomplete,
                                num_observed,
                            )
                            patterns.append([pattern])
                    else:
                        pattern = create_pattern(
                            incomplete_vars,
                            mechanism,
                            None,
                            score_to_probability_func,
                            num_incomplete,
                            None,
                        )
                        patterns.append([pattern])

    # Might have to move it do dev/scripts/ after
    with open("amputation_pattern_grid.txt", "w") as f:
        # json.dump(patterns, f)

        # To be able to pass to guild each patterns "list of dicts"
        # needs to be a string and each patterns entry needs to be comma separated with no whitespace
        f.write(
            json.dumps([str(pattern) for pattern in patterns], separators=(",", ":"))
        )

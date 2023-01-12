# !/bin/bash
num_queues=4

"""
These describe all the experiments that will be run with guild.
"""
percent_missing=[0.33,0.66]
feature_mapping=["onehot_categorical","target_encode_categorical","discretize_continuous"]
feature_mapping_variational=["target_encode_categorical"]

# baseline_imputers=["simple","mice", "knn"]
baseline_imputers=["simple","mice"]  
ae_imputers=["vanilla","dae","batchswap"]
vae_imputers=["vae","dvae"]
replace_nan_with=["simple","0"]

# experiment switches: all experiments: none, baseline, ae, vae
methods=( "none" "baseline" "ae" "vae" )
all_data=true # fully_observed=no uses entire dataset
fully_observed=false # fully_observed=yes will ampute and impute a missingness scenario


## Helper function ##
isin() {  # https://stackoverflow.com/a/3686056 the other responses didn't work
    value=$1
    array=$2
    for i in "${array[@]}"
    do
        if [[ "$i" -eq "$value" ]] ; then
            return 0
        fi
    done
    return 1
}

# https://my.guild.ai/t/command-run/146
# https://my.guild.ai/t/running-cases-in-parallel/341
# fully observed
if isin "none" "$methods"; then
    echo "======================================================"
    echo "Staging no imputation..."
    guild run train_predict --background method="none" fully-observed=yes
fi

if isin "baseline" "$methods"; then
    echo "======================================================"
    echo "Staging baseline imputers..."

    if [ $all_data = true ]; then
        # When multiple flags have list values, Guild generates the cartesian product of all possible flag combinations.
        guild run train_predict --background method=$baseline_imputers fully-observed=no
    fi

    if [ "$fully_observed" = true ]; then
        guild run train_predict --background method=$baseline_imputers fully-observed=yes percent-missing=$percent_missing amputation-patterns="$(cat dev/amputation_pattern_grid.txt)"
    fi
fi

if isin "ae" "$methods"; then
    echo "======================================================"
    echo "Staging AE imputers..."

    if [ "$all_data" = true ]; then
        guild run main --background method=$ae_imputers fully-observed=no feature-map=$feature_mapping replace-nan-with=$replace_nan_with
    fi
    if [ "$fully_observed" = true ]; then
        guild run main --background method=$ae_imputers fully-observed=yes feature-map=$feature_mapping percent-missing=$percent_missing replace-nan-with=$replace_nan_with amputation-patterns="$(cat dev/amputation_pattern_grid.txt)"
    fi
fi

if isin "vae" "$methods"; then
    echo "======================================================"
    echo "Staging VAE imputers..."

    # on vae and dvae only try target_encode_categorical
    if [ "$all_data" = true ]; then
        guild run main --background method=$vae_imputers fully-observed=no feature-map=$feature_mapping_variational replace-nan-with=$replace_nan_with 
    fi
    if [ "$fully_observed" = true ]; then
        guild run main --background method=$vae_imputers fully-observed=yes feature-map=$feature_mapping_variational percent-missing=$percent_missing replace-nan-with=$replace_nan_with amputation-patterns="$(cat dev/amputation_pattern_grid.txt)"
    fi
fi

# echo "======================================================"
# echo "Starting Queues..."

# for _ in `seq $num_queues`; do guild run queue --background -y; done
# guild view
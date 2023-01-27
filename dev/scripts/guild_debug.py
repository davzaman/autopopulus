from guild.commands.run import run

# Ref: https://my.guild.ai/t/debugging-guild-operations-in-pycharm/453/9
if __name__ == "__main__":
    run(
        [
            "main",
            # "train",
            "method=vae",
            "fully-observed=no",
            "feature-map=target_encode_categorical",
            "replace-nan-with=0",
            "--debug-sourcecode=.",
            "-y",
        ]
    )

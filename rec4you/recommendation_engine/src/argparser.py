import argparse

parser = argparse.ArgumentParser()


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


parser.add_argument(
    "--run",
    choices=[
        "train",
        "inference",
    ],
    help="Specify the run mode",
)

parser.add_argument(
    "--brand_ids",
    type=str,
    default="0,5,11,15,23,24,25,26,27,28,74,75,76,77,78,79",
    help="Comma-separated list of brand codes to run",
)

parser.add_argument("--n_months", type=int, default=48, help="Months back for the training dataset (transactions)")

parser.add_argument(
    "--website", type=str2bool, default=True, help="Run recommendations for website (True) or only email recs (False)"
)

parser.add_argument(
    "--type_of_recs",
    type=str,
    default="both",
    help="Select type of recommendations to run. Available options: 'personal', 'general', 'both'",
)

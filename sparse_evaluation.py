import os
from argparse import ArgumentParser

import pandas as pd

from evaluations.sparse_utils import sparsity_acc_test, sparsity_acc_test_lf_cbm

parser = ArgumentParser()
parser.add_argument("--load_path", type=str)
parser.add_argument("--lam", type=float, default=0.1)
parser.add_argument("--filter", type=float, default=0)
parser.add_argument("--annotation_dir", type=str, default=None)
parser.add_argument("--result_file", type=str, default=None)
parser.add_argument("--lf-cbm", action="store_true")

args = parser.parse_args()
if args.lf_cbm:
    accs = sparsity_acc_test_lf_cbm(args.load_path, lam_max=args.lam)
else:
    accs = sparsity_acc_test(
        args.load_path,
        lam_max=args.lam,
        bot_filter=args.filter,
        anno=args.annotation_dir,
    )
if args.result_file:
    if os.path.exists(args.result_file):
        df = pd.read_csv(args.result_file)
    else:
        df = pd.DataFrame(columns=["ACC@5", "AVGACC"])
    row = pd.Series({"ACC@5": accs[0], "AVGACC": sum(accs) / len(accs)})
    df.loc[len(df.index)] = row
    df.to_csv(args.result_file, index=False)

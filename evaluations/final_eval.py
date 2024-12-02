import os
import numpy as np
from argparse import ArgumentParser

from evaluations.sparse_utils import sparsity_acc_test

parser = ArgumentParser()
parser.add_argument("root", type=str)
parser.add_argument("--lam", type=float, default=0.1)
parser.add_argument("--annotation_dir", type=str, default=None)

args = parser.parse_args()
Acc5, AvgAcc = [], []
f = open(os.path.join(args.root, "Results.txt"), 'w')
for subdir in os.listdir(args.root):
    dir_path = os.path.join(args.root, subdir)
    if not os.path.isdir(dir_path): continue
    accs = sparsity_acc_test(dir_path, lam_max=args.lam,  anno=args.annotation_dir)
    Acc5.append(accs[0])
    AvgAcc.append(sum(accs) / len(accs))
    f.write(f"Accs: {accs}, AvgAcc: {sum(accs) / len(accs)}\n")
Acc5, AvgAcc = np.array(Acc5), np.array(AvgAcc)
f.write(f"Acc@5: Mean {Acc5.mean():.2%} Std: {Acc5.std():.2%}")
f.write(f"AvgAcc: Mean {AvgAcc.mean():.2%} Std: {AvgAcc.std():.2%}")
f.close()
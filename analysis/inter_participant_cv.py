"""
Performs inter-participant cross-validation on the data to determine the noise ceiling. 
Splits participants into random 50-50 splits and looks at correlation between scores
coming from the disjoint splits. 
Repeats this over 1000 random splits to estimate confidence intervals. 
"""

import pandas as pd
import numpy as np
import random
import argparse
from scipy.stats import spearmanr, pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
import os

CRITICAL = ["high", "mid", "low"]


def random_split(args):
    """Create random splits of the data and save parallel  dataframes where each row reprsents a different
    random 50-50 split of the data across participants and each column represents a sentence from the 
    critical conditions (1500 in total). Create three such pairs of matrices, one for each of accuracy, 
    false positive rate, and hit rate.

    Args:
        args (argparse namespace): command line args
    """
    df = pd.read_csv(args.csv)
    df = df[df.condition.isin(CRITICAL)]
    acc_data_1, acc_data_2 = [], []
    fpr_data_1, fpr_data_2 = [], []
    hr_data_1, hr_data_2 = [], []
    participants = list(set(df["participant"]))
    for i in range(args.num_splits):
        random.shuffle(participants)
        parts_1 = set(participants[: len(participants) // 2])
        parts_2 = set(participants[len(participants) // 2 :])

        # accuracy
        df1 = (df[df.participant.isin(parts_1)]
            .copy()
            .groupby("sentence")
            .agg({"correct": "mean"})
            .reset_index()
            .sort_values("sentence")
        )
        acc_data_1.append(dict(zip(df1.sentence, df1.correct)))

        df2 = (df[df.participant.isin(parts_2)]
            .copy()
            .groupby("sentence")
            .agg({"correct": "mean"})
            .reset_index()
            .sort_values("sentence")
        )
        acc_data_2.append(dict(zip(df2.sentence, df2.correct)))

        # fpr
        df1 = (df[df.participant.isin(parts_1) & ~df.repeat.astype(bool)]
            .copy()
            .groupby("sentence")
            .agg({"correct": "mean"})
            .reset_index()
            .sort_values("sentence")
        )
        fpr_data_1.append(dict(zip(df1.sentence, df1.correct)))

        df2 = (df[df.participant.isin(parts_2) & ~df.repeat.astype(bool)]
            .copy()
            .groupby("sentence")
            .agg({"correct": "mean"})
            .reset_index()
            .sort_values("sentence")
        )
        fpr_data_2.append(dict(zip(df2.sentence, df2.correct)))

        # hr
        df1 = (df[df.participant.isin(parts_1) & df.repeat.astype(bool)]
            .copy()
            .groupby("sentence")
            .agg({"correct": "mean"})
            .reset_index()
            .sort_values("sentence")
        )
        hr_data_1.append(dict(zip(df1.sentence, df1.correct)))

        df2 = (df[df.participant.isin(parts_2) & df.repeat.astype(bool)]
            .copy()
            .groupby("sentence")
            .agg({"correct": "mean"})
            .reset_index()
            .sort_values("sentence")
        )
        hr_data_2.append(dict(zip(df2.sentence, df2.correct)))

    acc_data_1 = pd.DataFrame(acc_data_1)
    acc_data_2 = pd.DataFrame(acc_data_2)
    acc_data_1.to_csv(args.acc_csv.replace(".csv", "_part1.csv"), index=False)
    acc_data_2.to_csv(args.acc_csv.replace(".csv", "_part2.csv"), index=False)

    fpr_data_1 = pd.DataFrame(fpr_data_1)
    fpr_data_2 = pd.DataFrame(fpr_data_2)
    fpr_data_1.to_csv(args.fpr_csv.replace(".csv", "_part1.csv"), index=False)
    fpr_data_2.to_csv(args.fpr_csv.replace(".csv", "_part2.csv"), index=False)

    hr_data_1 = pd.DataFrame(hr_data_1)
    hr_data_2 = pd.DataFrame(hr_data_2)
    hr_data_1.to_csv(args.hr_csv.replace(".csv", "_part1.csv"), index=False)
    hr_data_2.to_csv(args.hr_csv.replace(".csv", "_part2.csv"), index=False)


def correlations(args):

    ############################################################
    # accuracy 
    m1 = pd.read_csv(os.path.join(args.results_dir, args.acc_csv.replace(".csv", "_part1.csv")))
    m2 = pd.read_csv(os.path.join(args.results_dir, args.acc_csv.replace(".csv", "_part2.csv")))
    corrs = []

    for i in range(len(m1)):
        a1 = m1.iloc[i].to_numpy()
        a2 = m2.iloc[i].to_numpy()
        corr, p = spearmanr(a1, a2)
        # corr, p = pearsonr(a1, a2)
        corrs.append(corr)

    plt.clf()
    fig = sns.histplot(corrs)
    plt.savefig(os.path.join(args.results_dir, "img/acc_intersubject_cv"), dpi=args.dpi)
    print("Accuracy cross-validated Spearman correlations:")
    print(f"2.5, 50, 97.5 CIs: {np.percentile(corrs, 2.5):.2f} {np.percentile(corrs, 50):.2f} {np.percentile(corrs, 97.5):.2f}")
    print(pd.Series(corrs).describe(), "\n")

    ############################################################
    # fpr 
    m1 = pd.read_csv(os.path.join(args.results_dir, args.fpr_csv.replace(".csv", "_part1.csv")))
    m2 = pd.read_csv(os.path.join(args.results_dir, args.fpr_csv.replace(".csv", "_part2.csv")))
    corrs = []

    for i in range(len(m1)):
        a1 = m1.iloc[i].to_numpy()
        a2 = m2.iloc[i].to_numpy()
        corr, p = spearmanr(a1, a2)
        # corr, p = pearsonr(a1, a2)
        corrs.append(corr)

    plt.clf()
    fig = sns.histplot(corrs)
    plt.savefig(os.path.join(args.results_dir, "img/fpr_intersubject_cv"), dpi=args.dpi)
    print("False Alarm Rate cross-validated Spearman correlations:")
    print(f"2.5, 50, 97.5 CIs: {np.percentile(corrs, 2.5):.2f} {np.percentile(corrs, 50):.2f} {np.percentile(corrs, 97.5):.2f}")
    print(pd.Series(corrs).describe(), "\n")

    ############################################################
    # hr 
    m1 = pd.read_csv(os.path.join(args.results_dir, args.hr_csv.replace(".csv", "_part1.csv")))
    m2 = pd.read_csv(os.path.join(args.results_dir, args.hr_csv.replace(".csv", "_part2.csv")))
    corrs = []

    for i in range(len(m1)):
        a1 = m1.iloc[i].to_numpy()
        a2 = m2.iloc[i].to_numpy()
        corr, p = spearmanr(a1, a2)
        # corr, p = pearsonr(a1, a2)
        corrs.append(corr)

    plt.clf()
    fig = sns.histplot(corrs)
    plt.savefig(os.path.join(args.results_dir, "img/hr_intersubject_cv"), dpi=args.dpi)
    print("Hit Rate cross-validated Spearman correlations:")
    print(f"2.5, 50, 97.5 CIs: {np.percentile(corrs, 2.5):.2f} {np.percentile(corrs, 50):.2f} {np.percentile(corrs, 97.5):.2f}")
    print(pd.Series(corrs).describe(), "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv")
    parser.add_argument("--num_splits", type=int, default=1000)
    parser.add_argument("--random_split", action="store_true")
    parser.add_argument("--correlations", action="store_true")
    parser.add_argument("--acc_csv", default="acc_splits.csv")
    parser.add_argument("--fpr_csv", default="fpr_splits.csv")
    parser.add_argument("--hr_csv", default="hr_splits.csv")
    parser.add_argument("--results_dir", default="interparticipant")
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()

    if args.random_split:
        random_split(args)
    if args.correlations:
        correlations(args)


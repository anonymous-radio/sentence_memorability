"""
This script calculates the basic performance metrics - accuracy, hit rate, and false alarm rate - 
on the experimental data and computes bootstrapped 95% CI intervals. 
Generates a barplot of the metrics. 
"""

import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter

sns.set(font_scale=2)
sns.set_style("white")
plt.rcParams['xtick.bottom'] = True
plt.rcParams['ytick.left'] = True

CSV_FILE = "../data/sent-mem-data.csv"
NUM_SPLITS = 1000
NUM_SAMPLES = 750
TARGETS = ["high", "mid", "low"]

if __name__ == "__main__":
    """Split data into many random splits, take half of data and compute median of this subsample, and then
    take the 2.5, 50, and 97.5 percentiles of this distribution. This estimates a median and 95% CI for 
    each of the metrics - accuracy, false alarm rate, and hit rate. 
    """
    df = pd.read_csv(CSV_FILE).melt(id_vars=["sentence"], value_vars=["target_hr", "fpr", "acc"], var_name="metric", value_name="value")

    median_performances = []
    low_performances = []
    high_performances = []
    for metric in ["acc", "target_hr", "fpr"]:
        medians = []
        for i in tqdm(range(NUM_SPLITS)):
            sub = df[df.metric == metric].sample(NUM_SAMPLES)["value"]
            medians.append(np.percentile(sub, 50))
        # print(metric, np.percentile(medians, [2.5, 50, 97.5]))
        median_performances.append(np.percentile(medians, 50.0))
        low_performances.append(np.percentile(medians, 2.5))
        high_performances.append(np.percentile(medians, 97.5))

    metrics = ["accuracy", "hit rate", "false alarm rate"]
    df = pd.DataFrame({"metric": metrics, "median performance": median_performances, "low": low_performances, "high": high_performances})

    # filter and sort
    df = df[df.metric.isin(metrics)]
    df = df.sort_values("metric", key=lambda x: x.replace(dict(zip(metrics, range(len(metrics))))))

    # make the plot
    color_mapping = {}
    f, ax = plt.subplots(figsize=(9, 7))
    fig = sns.barplot(data=df, x="metric", y="median performance", hue="metric", dodge=False, alpha=0.5, palette="gray", order=metrics)
    
    # add error bars
    error = sns.utils.ci_to_errsize((df["low"], df["high"]), df["median performance"])
    ax.errorbar(df["metric"], df["median performance"], yerr=error, ecolor='k', elinewidth=2, fmt="none")
    # ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.yaxis.set_major_formatter(FuncFormatter('{0:.0%}'.format))
    ax.set(xlabel=None)
    ax.get_legend().remove()
    
    # axes, title, and legend
    plt.title("Memorability Performance of Sentences")

    # save files
    plt.savefig(f"img/sentence_metrics", dpi=180, bbox_inches="tight")
    plt.savefig(f"img/sentence_metrics.svg", dpi=180, bbox_inches="tight")
    plt.savefig(f"img/sentence_metrics.pdf", dpi=180, bbox_inches="tight")



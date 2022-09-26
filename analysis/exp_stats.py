"""
This program perform basic statistical model fitting and model comparison.
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

TARGETS = ["high", "mid", "low"]

if __name__ == "__main__":
    df = pd.read_csv("../data/sent-mem-data.csv")
    df_sub = (df[df.condition.isin(TARGETS)]
        .copy()
        .groupby("sentence")
        .mean()
        .reset_index()
        .sort_values("sentence")
    )

    df2 = pd.read_csv("../embeddings/critical_acc_v_avg_dist_sbert.csv")
    df_sub = df_sub.merge(df2, on="sentence")

    # demeaning
    df_sub["acc"] = df_sub["acc"] - df_sub["acc"].mean()
    df_sub["max_mem"] = df_sub["max_mem"] - df_sub["max_mem"].mean()
    df_sub["avg_dist_sbert"] = df_sub["avg_dist_sbert"] - df_sub["avg_dist_sbert"].mean()

    # fit linear regressions for max mem and max_mem + sbert distinctiveness
    m1 = smf.ols('acc ~ max_mem', data=df_sub).fit()
    m2 = smf.ols('acc ~ max_mem + avg_dist_sbert', data=df_sub).fit()

    # do anova and print
    comparison = sm.stats.anova_lm(m1, m2)
    print(comparison)



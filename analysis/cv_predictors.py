"""
This program performs analyses and visualizations related to cross-validation and predictivity
of various predictors for sentence memorability. 
"""

import pandas as pd
import numpy as np
import utils
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import glob

def fit(args):
    """Apply the get_cv_score() function to given data and predictors.
    df_csv should contain the experimental data with a column for the desired predictor
    y1_csv and y2_csv should contain the CV results across disjoint splits of participants,
        which can be based on accuracy, false alarm rate, or hit rate

    Args:
        args (argparse.NameSpace): command line args

    Raises:
        ValueError: raises an error if the input dataframe is empty
    """
    df = pd.read_csv(args.df_csv)
    acc1 = pd.read_csv(args.y1_csv)
    acc2 = pd.read_csv(args.y2_csv)

    if args.restore_stimset:
        materials = pd.read_csv(args.materials_csv)
        mapping = dict(zip(materials.sentence, materials.condition))
        df["condition"] = df["sentence"].apply(lambda x: mapping.get(x))
        assert df["condition"].notna().all()

    df_sub = df[df.condition.isin(["high", "mid", "low"])]
    df_sub = df_sub.groupby("sentence").mean().reset_index()
    df_sub = df_sub.sort_values(by="sentence")

    if len(df_sub) <= 0:
        raise ValueError("The input dataframe was empty.")

    df_save = utils.get_cv_score(
        df_sub,
        acc1, 
        acc2, 
        predictors = [str(args.predictor)],
        model_name=args.model_name, 
        result_dir="cv-results", 
        save_subfolder=args.save_subfolder,
    )

def fit_multiple(args):
    """Similar to fit() function, but with multiple predictors

    Args:
        args (argparse.NameSpace): command line args

    Raises:
        ValueError: raises exception if the input dataframe is empty
    """
    df1 = pd.read_csv(args.df1_csv)
    df2 = pd.read_csv(args.df2_csv)
    acc1 = pd.read_csv(args.y1_csv)
    acc2 = pd.read_csv(args.y2_csv)

    df = df1.merge(df2, on="sentence")
    predictors = args.predictor.split(",")

    if args.restore_stimset:
        materials = pd.read_csv(args.materials_csv)
        mapping = dict(zip(materials.sentence, materials.condition))
        df["condition"] = df["sentence"].apply(lambda x: mapping.get(x))
        assert df["condition"].notna().all()

    df_sub = df[df.condition.isin(["high", "mid", "low"])]
    df_sub = df_sub.groupby("sentence").mean().reset_index()
    df_sub = df_sub.sort_values(by="sentence")

    if len(df_sub) <= 0:
        raise ValueError("The input dataframe was empty.")

    df_save = utils.get_cv_score(
        df_sub,
        acc1, 
        acc2, 
        predictors=predictors,
        model_name="_v_".join(predictors), 
        result_dir="cv-results", 
        save_subfolder=args.save_subfolder,
    )

def fit_embs(args):
    """Fit a regression predicting the outcome based on entire embeddings.

    Args:
        args (argparse.NameSpace): command line arguments

    Raises:
        ValueError: raise exception if input dataframe is empty
    """
    acc1 = pd.read_csv(args.y1_csv)
    acc2 = pd.read_csv(args.y2_csv)

    actv = pd.read_pickle(f"../model-actv/{args.embedding_model}/{args.sent_emb}/sentmem_actv.pkl")
    stim = pd.read_pickle(f"../model-actv/{args.embedding_model}/{args.sent_emb}/sentmem_stim.pkl")

    actv = actv[args.embedding_model_layer]
    actv["sentence"] = stim["sentence"]
    actv["condition"] = stim["condition"]

    df_sub = actv[actv.condition.isin(["high", "mid", "low"])]
    df_sub = df_sub.sort_values(by="sentence")

    if len(df_sub) <= 0:
        raise ValueError("The input dataframe was empty.")

    df_save = utils.get_cv_score(
        df_sub,
        acc1, 
        acc2, 
        predictors = df_sub.columns[:-2],
        model_name=f"{args.embedding_model}_{args.sent_emb}_layer-{str(args.embedding_model_layer)}", 
        result_dir="cv-results", 
        save_subfolder="embs"
    )

def reformat_layer_num(s):
    """Apply formatting change so that layers will appear in numerical order.

    Args:
        s (str): a string

    Returns:
        str: adds zero padding to numerical parts of strings to maintain sorting behavior
    """
    if not "-" in s:
        return s
    parts = s.split("-")
    num = parts[-1]
    num = num.zfill(2)
    return "-".join(parts[:-1]) + "-" + num

def reformat_layer_num_numeric(s):
    """Extract numerical layer number from predictor identifier

    Args:
        s (str): a string

    Returns:
        int: layer number (last int in input string)
    """
    if not "-" in s:
        return s
    parts = s.split("-")
    num = parts[-1]
    return int(num)

def viz(args):
    """Make and save plot.

    Args:
        args (argparse.NameSpace): command line args
    """
    # matplotlib settings
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['svg.fonttype'] = 'none'

    # read in the data
    filenames = glob.glob(f"cv-results/{args.save_subfolder}/cv_summary_preds/*0925.csv")
    df = pd.concat([pd.read_csv(x) for x in filenames])
    col_mapping = {
        df.columns[0]: "predictor", 
        f"lower_CI2.5_{args.corr_method}": "CI2.5", 
        f"median_CI50_{args.corr_method}": "CI50", 
        f"upper_CI97.5_{args.corr_method}": "CI97.5"
    }
    df = df.rename(columns=col_mapping)
    df["predictor"] = df["predictor"].apply(reformat_layer_num)
    
    # assign category names
    df["category"] = ""
    df.loc[df.predictor.str.contains("bert-large-cased"), "category"] = "bert"
    df.loc[df.predictor.str.contains("gpt2-medium"), "category"] = "gpt"
    df.loc[df.predictor.str.contains("electra"), "category"] = "sentence emb."
    df.loc[df.predictor.str.contains("sbert"), "category"] = "sentence emb."
    df.loc[df.predictor.str.contains("use"), "category"] = "sentence emb."
    df.loc[df.predictor.str.contains("mem"), "category"] = "lexical-model"
    df.loc[df.predictor.str.contains("glove"), "category"] = "lexical"
    df.loc[df.predictor.str.contains("resnet"), "category"] = "visual"
    df.loc[df.predictor.str.contains("_v_"), "category"] = "joint"

    # filter and sort
    bar_order = [
        "avg_dist_resnet50-early", 
        "mean_mem", 
        "max_mem",
        "avg_dist_glove", 
        "avg_dist_electra", 
        "avg_dist_use", 
        "avg_dist_sbert", 
        "avg_dist_bert-large-cased_layer-09", 
        "avg_dist_gpt2-medium_layer-23", 
        "avg_dist_sbert_v_max_mem"
    ]
    df = df[df.predictor.isin(bar_order)]
    df = df.sort_values("predictor", key=lambda x: x.replace(dict(zip(bar_order, range(len(bar_order))))))

    # make the plot
    color_mapping = {
        "bert": "tab:orange", 
        "gpt": "tab:blue", 
        "sentence emb.": "green", 
        "lexical-model": "red", 
        "lexical": "pink", 
        "visual": "gray", 
        "joint": "purple"
    }
    f, ax = plt.subplots(figsize=(9, 5))
    fig = sns.barplot(data=df, x="predictor", y="CI50", hue="category", dodge=False, alpha=0.5, palette=color_mapping, order=bar_order)
    
    # add error bars
    error = sns.utils.ci_to_errsize((df["CI2.5"], df["CI97.5"]), df["CI50"])
    ax.errorbar(df["predictor"], df["CI50"], yerr=error, ecolor='k', elinewidth=2, fmt="none")

    # add noise ceiling line and confidence interval
    ci_low, ci_med, ci_high = 0.52, 0.56, 0.60
    plt.axhline(y=ci_med, color='black', linestyle='--', label="noise ceiling")
    plt.axhspan(ci_low, ci_high, alpha=0.3, color='gray', zorder=0)
    
    # axes, title, and legend
    plt.xticks(rotation=30)
    plt.title("Cross-Validation Predictivity of \nLexical and Sentence Features on Memorability")
    plt.legend(loc="upper right", bbox_to_anchor=(1.25, 1.0))

    # save files
    plt.savefig(f"cv-results/{args.save_subfolder}/{args.save_subfolder}_predictivity", dpi=180, bbox_inches="tight")
    plt.savefig(f"cv-results/{args.save_subfolder}/{args.save_subfolder}_predictivity.svg", dpi=180, bbox_inches="tight")
    plt.savefig(f"cv-results/{args.save_subfolder}/{args.save_subfolder}_predictivity.pdf", dpi=180, bbox_inches="tight")

def viz_embs(args):
    filenames = glob.glob(f"cv-results/embs/cv_summary_preds/*.csv")
    df = pd.concat([pd.read_csv(x) for x in filenames])
    col_mapping = {
        df.columns[0]: "predictor", 
        f"lower_CI2.5_{args.corr_method}": "CI2.5", 
        f"median_CI50_{args.corr_method}": "CI50", 
        f"upper_CI97.5_{args.corr_method}": "CI97.5"
    }
    df = df.rename(columns=col_mapping)
    df["predictor"] = df["predictor"].apply(reformat_layer_num)
    df["category"] = ""
    df.loc[df.predictor.str.contains("bert"), "category"] = "bert"
    df.loc[df.predictor.str.contains("gpt"), "category"] = "gpt"
    df = df.sort_values(by=["category", "predictor"])
    f, ax = plt.subplots(figsize=(9, 5))
    fig = sns.barplot(data=df, y="predictor", x="CI50", hue="category", dodge=False)
    error = sns.utils.ci_to_errsize((df["CI2.5"], df["CI97.5"]), df["CI50"])
    ax.errorbar(df["CI50"], df["predictor"], xerr=error, ecolor='k', elinewidth=2, fmt="o")
    plt.title("Cross-Validation Predictivity of \n Transformer Activations on Memorability")
    plt.savefig("cv-results/run1/predictivity_from_embs", dpi=150, bbox_inches="tight")

def viz_layers(args):
    """Make plot for layer-by-layer predictivity.

    Args:
        args (argparse.NameSpace): command line args
    """

    # matplotlib settings
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['svg.fonttype'] = 'none'

    # read in data
    filenames = (glob.glob(f"cv-results/{args.save_subfolder}/cv_summary_preds/*bert-large-cased*0925.csv") + 
        glob.glob(f"cv-results/{args.save_subfolder}/cv_summary_preds/*gpt2-medium*0925.csv"))
    df = pd.concat([pd.read_csv(x) for x in filenames])
    col_mapping = {
        df.columns[0]: "predictor", 
        f"lower_CI2.5_{args.corr_method}": "CI2.5", 
        f"median_CI50_{args.corr_method}": "CI50", 
        f"upper_CI97.5_{args.corr_method}": "CI97.5"
    }
    df = df.rename(columns=col_mapping)

    # assign category labels
    df["category"] = ""
    df.loc[df.predictor.str.contains("bert"), "category"] = "bert"
    df.loc[df.predictor.str.contains("gpt"), "category"] = "gpt"
    
    # reformat predictor and sort
    df["predictor"] = df["predictor"].apply(reformat_layer_num_numeric)
    df = df.sort_values(by=["category", "predictor"])

    # make barplot
    plt.figure(figsize=(8,4))
    fig = sns.barplot(data=df, x="predictor", y="CI50", hue="category", dodge=False, alpha=0.5, hue_order=["gpt", "bert"])

    # error bars
    df_bert = df[df.category == "bert"]
    error = sns.utils.ci_to_errsize((df_bert["CI2.5"], df_bert["CI97.5"]), df_bert["CI50"])
    plt.errorbar(df_bert["predictor"], df_bert["CI50"], yerr=error, ecolor='orangered', elinewidth=1, alpha=0.5, fmt="none")
    df_gpt = df[df.category == "gpt"]
    error = sns.utils.ci_to_errsize((df_gpt["CI2.5"], df_gpt["CI97.5"]), df_gpt["CI50"])
    plt.errorbar(df_gpt["predictor"], df_gpt["CI50"], yerr=error, ecolor='navy', elinewidth=1, alpha=0.5, fmt="none")

    # noise ceiling
    plt.axhline(y=0.56, color='black', linestyle='--', label="noise ceiling")
    plt.axhspan(0.52, 0.60, alpha=0.3, color='gray', zorder=0)

    # title, legend, save
    plt.title("Cross-Validation Predictivity of \n Transformer Activation Distinctiveness")
    plt.legend(loc="upper left", bbox_to_anchor=(0, 0.8))
    plt.savefig(f"cv-results/{args.save_subfolder}/{args.save_subfolder}_predictivity_by_layer", dpi=180, bbox_inches="tight")
    plt.savefig(f"cv-results/{args.save_subfolder}/{args.save_subfolder}_predictivity_by_layer.svg", dpi=180, bbox_inches="tight")
    plt.savefig(f"cv-results/{args.save_subfolder}/{args.save_subfolder}_predictivity_by_layer.pdf", dpi=180, bbox_inches="tight")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--materials_csv", default="materials_w-sent-identifier.csv")
    parser.add_argument("--df_csv", default="../data/sent-mem-data.csv")
    parser.add_argument("--df1_csv", default="../embeddings/critical_acc_v_avg_dist_sbert.csv")
    parser.add_argument("--df2_csv", default="../data/sent-mem-data.csv")
    parser.add_argument("--y1_csv", default="interparticipant/acc_splits_part1.csv")
    parser.add_argument("--y2_csv", default="interparticipant/acc_splits_part2.csv")
    parser.add_argument("--predictor", default="mean_mem")
    parser.add_argument("--model_name")
    parser.add_argument("--fit", action="store_true")
    parser.add_argument("--fit_embs", action="store_true")
    parser.add_argument("--fit_multiple", action="store_true")
    parser.add_argument("--viz", action="store_true")
    parser.add_argument("--viz_embs", action="store_true")
    parser.add_argument("--viz_layers", action="store_true")
    parser.add_argument("--restore_stimset", action="store_true")
    parser.add_argument("--corr_method", default="spearman")
    parser.add_argument("--embedding_model", default="gpt2-xl")
    parser.add_argument("--embedding_model_layer", type=int, default=11)
    parser.add_argument("--sent_emb", default="last-tok")
    parser.add_argument("--save_subfolder", default="acc")
    args = parser.parse_args()

    if not args.corr_method in ["spearman", "pearson"]:
        raise ValueError("Correlation method not supported.")

    if not args.model_name:
        args.model_name = args.predictor    

    if args.fit:
        fit(args)
    if args.fit_multiple:
        fit_multiple(args)
    if args.fit_embs:
        fit_embs(args)
    if args.viz:
        viz(args)
    if args.viz_embs:
        viz_embs(args)
    if args.viz_layers:
        viz_layers(args)
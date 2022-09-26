# INFO
# Input: CSV files of sentences that have undergone manual review, should have "reject2"
#           containing '1' or '2' if the sentence should be rejected
# Output: .js files containing stimuli lists
# Side effect: new CSV containing columns 'has_edit' and 'final_sentence'

import argparse
import pandas as pd
import numpy as np
import os


def incorporate_manual_review(df, filename, keep_lines, strategy="reject2"):
    df = df[df["reject"] != 1].copy()

    # copy the original sentences
    sentences = df["sentence"].copy()

    # get mask of where there was an edit
    edit_mask = ~df["sentence_edit"].isna()

    # update the edited sentences
    sentences.loc[edit_mask] = df["sentence_edit"]

    # create new column for 'has_edit'
    df["has_edit"] = sentences != df["sentence"]

    # sentence_edit should contain the original sent if no edit, edited sent otherwise
    df["sentence_edit"] = sentences
    # df["sentence_edit"] = sentences[df["has_edit"]]
    # df["sentence"].loc[df["has_edit"]] = df["sentence_edit"]

    # rename the columns:
    df = df.rename(
        columns={"sentence": "sentence_original", "sentence_edit": "sentence"}
    )
    df.to_csv(filename.replace(".csv", "_filt.csv"), index=False)

    return df


def make_js_file(df, num_lists, varname, filename):

    sub_dfs = np.array_split(df, num_lists)

    lines = [f"var {varname} = ["]
    for sub_df in sub_dfs:
        lines.append("[")
        for sent in sub_df["sentence"]:
            sent_esc = sent.replace('"', '\\"')
            lines.append(f'"{sent_esc}",')
        lines.append("],")
    lines.append("];\n\n")

    with open(filename, "a") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--high", action="store_true")
    parser.add_argument("--mid", action="store_true")
    parser.add_argument("--low", action="store_true")
    parser.add_argument("--misc", action="store_true")
    parser.add_argument("--filler", action="store_true")
    parser.add_argument("--num_lists", type=int, default=12)
    args = parser.parse_args()

    if args.high:
        in_filename = "sentences_verified/sentences_high.csv"
        out_filename = "../exp/static/data/data_high.js"

        if os.path.exists(out_filename):
            os.remove(out_filename)

        df = pd.read_csv(in_filename)
        df = incorporate_manual_review(df, in_filename, keep_lines=None)
        df = df.sample(n=min(500, len(df)))

        make_js_file(df, args.num_lists, "sents_high", out_filename)

    if args.mid:
        in_filename = "sentences_verified/sentences_mid.csv"
        out_filename = "../exp/static/data/data_mid.js"

        if os.path.exists(out_filename):
            os.remove(out_filename)

        df = pd.read_csv(in_filename)
        df = incorporate_manual_review(df, in_filename, keep_lines=None)
        df = df.sample(n=min(500, len(df)))

        make_js_file(df, args.num_lists, "sents_mid", out_filename)

    if args.low:
        in_filename = "sentences_verified/sentences_low.csv"
        out_filename = "../exp/static/data/data_low.js"

        if os.path.exists(out_filename):
            os.remove(out_filename)

        df = pd.read_csv(in_filename)
        df = incorporate_manual_review(df, in_filename, keep_lines=None)
        df = df.sample(n=min(500, len(df)))

        make_js_file(df, args.num_lists, "sents_low", out_filename)

    if args.misc:
        in_filename = "../misc-sentences.csv"
        out_filename = "../exp/static/data/data_misc.js"

        if os.path.exists(out_filename):
            os.remove(out_filename)

        df = pd.read_csv(in_filename)
        df = df.sample(frac=1)

        make_js_file(df, args.num_lists, "sents_misc", out_filename)

    if args.filler:
        in_filename = "sentences_verified/sentences_filler.csv"
        out_filename = "../exp/static/data/data_filler.js"

        if os.path.exists(out_filename):
            os.remove(out_filename)

        df = pd.read_csv(in_filename)
        df = incorporate_manual_review(df, in_filename, keep_lines=None)
        df = df.sample(n=min(1000, len(df)))

        make_js_file(df, 1, "sents_filler", out_filename)

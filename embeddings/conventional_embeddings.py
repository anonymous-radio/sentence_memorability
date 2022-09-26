import fasttext
import fasttext.util
import gensim
import gensim.downloader
import argparse
import pandas as pd
import numpy as np
from glob import glob
from sklearn import linear_model
import embedding_utils as utils
import re
import itertools


def calculate_memorability_conventional(regr, sent_csv, args):
    """Read sentences from CSV and calculate pseudo-memorability scores using
    trained linear regression model on word embeddings.

    Args:
        regr (object): trained linear regression model
        sent_csv (str): filepath to CSV with sentences
        args (object): additional arguments

    Returns:
        object: dictionary containing embeddings, memorabilities, and words
    """

    df = pd.read_csv(sent_csv)
    sentences = map(lambda x: re.sub(r"[^\w\s]", "", x), df["sentence"])
    all_words = list(set(" ".join(sentences).lower().split()))

    embeddings, oov_words = get_embeddings(args, all_words)

    # use activations to predict accuracy (memorability)
    accuracies = regr.predict(embeddings)

    # word_acc = list(zip(all_words, accuracies, embeddings))
    # word_acc = sorted(word_acc, key=lambda x: x[1])
    # word_accs["activations"] = [pair[2] for pair in word_acc]
    # word_accs["accuracies"] = [pair[1] for pair in word_acc]
    # word_accs["words"] = [pair[0] for pair in word_acc]

    data = {}
    data["activations"] = embeddings
    data["accuracies"] = accuracies
    data["words"] = all_words
    data["oov_words"] = oov_words
    return data


def get_embeddings(args, words):
    if args.embedding_method == "fasttext":
        return utils.get_embeddings_ft(words)
    if args.embedding_method == "glove":
        return utils.get_embeddings_glove(words, embs_file=args.glove_file)
    if args.embedding_method == "word2vec":
        return utils.get_embeddings_word2vec(words)


def run_regression(args, embeddings, accuracies):
    # train linear regression
    regr = linear_model.LinearRegression()
    scores, _, _ = utils.cross_validation(
        embeddings, accuracies, regr, k=args.num_splits, group_values=embeddings
    )
    print("Cross-Validated Regression Scores:", scores)
    print("Mean Score:", np.mean(scores))
    utils.save_linear_model(regr, f"linear_models/{args.embedding_method}.pkl")
    return regr


def estimate_memorability(args):
    regr = utils.read_linear_model(f"linear_models/{args.embedding_method}.pkl")

    data = calculate_memorability_conventional(
        regr, sent_csv=args.sentences_file, args=args
    )

    if args.dim_reduc_method == "pca":
        dim_reduc = utils.do_pca
    elif args.dim_reduc_method == "tsne":
        dim_reduc = utils.do_tsne
    data["X"] = dim_reduc(
        X_in=data["activations"], n_components=args.dim_reduc_components
    )

    utils.plot_accs(
        accuracies=data["accuracies"],
        words=data["words"],
        X=data["X"][:],
        title="Word Embeddings (Dimensionality Reduction)",
        show_fig=False,
        save_fig=True,
        file_name=f"img/pseudo_memorability_{args.embedding_method}.png",
    )

    df = pd.DataFrame({"word": data["words"], "accuracy": data["accuracies"]})
    df["is_oov"] = df["word"].apply(lambda x: x in data["oov_words"])
    df = df.sort_values("accuracy")
    df.to_csv(
        f"pseudo_memorabilities/pseudo_memorability_{args.embedding_method}.csv",
        index=False,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--uppercase", action="store_true")
    parser.add_argument(
        "--embedding_method", default="glove", help="one of fastttext or glove"
    )
    parser.add_argument("--num_splits", type=int, default=10)
    parser.add_argument(
        "--sentences_file", default="../in/brown/brown_6_stimuli_memorability.csv"
    )
    parser.add_argument(
        "--memorability_file_1",
        default="../exp1_data_with_norms_reordered_gt_20211220.csv",
    )
    parser.add_argument(
        "--memorability_file_2",
        default="../exp2_data_with_norms_reordered_gt_20211220.csv",
    )
    parser.add_argument("--dim_reduc_method", help="pca or tsne", default="pca")
    parser.add_argument("--dim_reduc_components", type=int, default=2)
    parser.add_argument("--glove_file", default="glove.42B.300d.txt")
    args = parser.parse_args()

    # get words with memorability scores
    df1 = pd.read_csv(args.memorability_file_1)
    df2 = pd.read_csv(args.memorability_file_2)
    df2 = df2[df2.multi_word == 0]
    df = pd.concat([df1, df2])

    # df["wordcount"] = df["word_lower"].apply(lambda x: len(x.split()))
    # df = df[df.wordcount == 1]

    oldlen = len(df)
    df = df.drop_duplicates("word_lower")
    print(f"Old length: {oldlen}\nNew length: {len(df)}")

    words = list(df.word_upper) if args.uppercase else list(df.word_lower)
    accuracies = np.array(df.acc)

    embeddings, oov_words = get_embeddings(args, words)
    print("OOV words:", oov_words)
    print("len OOV words:", len(oov_words))
    oov_set = set(oov_words)

    words_series = pd.Series(words)
    is_oov = words_series.isin(oov_set)

    embeddings = embeddings[~is_oov]
    accuracies = accuracies[~is_oov]

    print("Shape of Embeddings:", embeddings.shape)
    run_regression(args, embeddings, accuracies)
    estimate_memorability(args)

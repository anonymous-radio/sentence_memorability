from collections import defaultdict
import torch
import warnings
import numpy as np
import pandas as pd
import re
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
# import plotly.express as px
# from plotly.io import write_image
import matplotlib.pyplot as plt
from transformers import GPT2Tokenizer, GPT2Model
import pickle
# import fasttext, fasttext.util
# import gensim, gensim.downloader


def get_embeddings_ft(words):
    """Get fasttext word embeddings

    Args:
        words (List[str]): list of words

    Returns:
        Tuple: tuple of fasttext embedding matrix and word list
    """
    fasttext.util.download_model("en", if_exists="ignore")
    embs = fasttext.load_model("cc.en.300.bin")
    a = []
    oov_words = []
    for word in words:
        if word in embs:
            a.append(embs[word])
        else:
            a.append(np.full(300, 0))
            oov_words.append(word)
    return np.array(a), oov_words


def read_glove_vectors_from_file(filename):
    d = {}
    with open(filename, "rt") as f:
        for line in f:
            word, *rest = line.split()
            d[word] = np.array(list(map(float, rest)))
    return d


def get_embeddings_glove(words, embs_file):
    embs = read_glove_vectors_from_file(embs_file)
    a = []
    oov_words = []
    for word in words:
        try:
            vec = embs[word]
            a.append(vec)
        except KeyError:
            a.append(np.full(300, 0))
            oov_words.append(word)
    return np.array(a), oov_words


def get_embeddings_word2vec(words):
    model = gensim.downloader.load("word2vec-google-news-300")
    a = []
    oov_words = []
    for word in words:
        try:
            vec = model[word]
            a.append(vec)
        except KeyError:
            a.append(np.full(300, 0))
            oov_words.append(word)
    return np.array(a), oov_words


def calculate_memorability(
    layer_regressions, model_name, sent_csv, args, dict_key="layer_{}",
):
    """
    function uses gpt2 to calculate memorability

    Input(s)
    --------
    linear_models : the linear models trained for each layer
    """

    # now we need to get accuracy values for all words
    model = GPT2Model.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    # n_layers = model.config.n_layer
    n_layers = args.num_layers
    print("n_layers: ", n_layers)

    # gather all linear models
    # regressors = [
    #     layer_scores[dict_key.format(n_layer)]["linear_model"]
    #     for n_layer in range(n_layers)
    # ]

    df = pd.read_csv(sent_csv)

    # look at a sentence, clean it up, extract words, add to set
    sentences = map(lambda x: re.sub(r"[^\w\s]", "", x), df["sentence"])
    all_words = list(set(" ".join(sentences).lower().split()))

    # get activations from gpt
    activations = get_activations(
        model=model,
        tokenizer=tokenizer,
        sents=all_words,
        sentence_embedding="avg-tok",
        verbose=True,
    )

    word_accs = defaultdict(dict)

    for n_layer in range(n_layers):
        # get the linear model for the layer
        layer_model = layer_regressions[n_layer]

        # calculate the actiations
        layer_activations = activations[n_layer]

        # use activations to predict accuracy (memorability)
        accuracies = layer_model.predict(layer_activations)

        word_acc = list(zip(all_words, accuracies, layer_activations))
        word_acc = sorted(word_acc, key=lambda x: x[1])

        word_accs[dict_key.format(n_layer)]["activations"] = np.array(
            [pair[2] for pair in word_acc]
        )
        word_accs[dict_key.format(n_layer)]["accuracies"] = np.array(
            [pair[1] for pair in word_acc]
        )
        word_accs[dict_key.format(n_layer)]["words"] = [pair[0] for pair in word_acc]

    return word_accs


def get_activations(
    model, tokenizer, sents, sentence_embedding, lower=True, verbose=True
):
    """
    :param sents: list of strings
    :param sentence_embedding: string denoting how to obtain sentence embedding

    hidden_states (in Transformers==4.1.0) is a 3D tensor of dims: [batch, tokens, emb size]

    Compute activations of hidden units
    Returns dict with key = layer, item = 2D array of stimuli x units
    """

    model.eval()  # does not make a difference
    n_layer = model.config.n_layer
    max_n_tokens = model.config.n_positions
    states_sentences = defaultdict(list)
    if verbose:
        print(f"Computing activations for {len(sents)} sentences")

    for count, sent in enumerate(sents):
        if lower:
            sent = sent.lower()
        input_ids = torch.tensor(tokenizer.encode(sent))

        start = max(0, len(input_ids) - max_n_tokens)
        if start > 0:
            warnings.warn(f"Stimulus too long! Truncated the first {start} tokens")
        input_ids = input_ids[start:]
        result_model = model(input_ids, output_hidden_states=True, return_dict=True)
        hidden_states = result_model["hidden_states"]

        #### NOTE:
        # hidden_state[n_layer][word][emb for token]
        for i in range(n_layer):  # for each layer
            dim = 0
            state = None
            hidden_state_shape = hidden_states[i].shape
            if hidden_state_shape[1] <= 1:
                state = hidden_states[i].squeeze().detach().numpy()
            elif sentence_embedding == "last-tok":  # last token
                state = hidden_states[i].squeeze()[-1, :].detach().numpy()
            elif sentence_embedding == "avg-tok":  # mean over tokens
                state = torch.mean(hidden_states[i].squeeze(), dim=dim).detach().numpy()
            elif sentence_embedding == "sum-tok":  # sum over tokens
                state = torch.sum(hidden_states[i].squeeze(), dim=dim).detach().numpy()
            else:
                print("Sentence embedding method not implemented")
            states_sentences[i].append(np.array(state))
    return states_sentences


def cross_validation(X, y, initialized_model, k, group_values=None):
    """
    Computes cross-validation, k-fold

    :param X: source
    :param y: target
    :param initialized_model: mapping model
    :param group_values: the unique stimuli
    :param k:
    :param special:
    :param target_col:
    :return:
    """
    # groups = np.unique(group_values, axis=0)
    # print("shape(groups)", groups.shape)
    kf = KFold(n_splits=k, shuffle=True, random_state=0)
    model = initialized_model

    ## Store ##
    r_across_splits = []
    train_indices = []
    test_indices = []

    for train_index, test_index in kf.split(X):
        # for train_index, test_index in kf.split(groups):
        # train_index = [
        #     i for i, group in enumerate(group_values) if group in groups[train_index]
        # ]
        # test_index = [i for i in range(len(X)) if i not in train_index]
        # train_indices.append(train_index)
        # test_indices.append(test_index)

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_pred = np.squeeze(y_pred)
        r_across_splits.append(np.corrcoef(y_pred, y_test)[0][1])

    print(f"Mean cross-validation score over {k} folds: {np.mean(r_across_splits):.3f}")

    # fit model on full data (not just 90%)
    model.fit(X, y)

    return r_across_splits, train_indices, test_indices


def activations_to_csv(activations, sents, acc, n_layers=0, filename=""):
    """
    'ActivationsToCSV' grabs activations from 'get_activations' function and
    saves them to a CSV file of name 'filename'.

    Each row will have a word and corresponding activations at each layer (layer number will be column)
    """
    # embedding for word in words = activations[n_layer][idx(word)]

    # get column names for new dataframe (to save to CSV)
    layer_str = "layer_{}_activations"
    column_names = ["word_upper", "word_lower", "acc"]
    column_names.extend([layer_str.format(layer) for layer in range(len(activations))])
    df = pd.DataFrame(columns=column_names)

    for idx in range(len(sents)):
        # get current word or set of tokens
        curr_word = sents[idx]
        row_to_add = [curr_word.upper(), curr_word.lower(), acc[idx]]

        # get activations for current word at every layer
        for n_layer in range(n_layers):
            curr_activations = activations[n_layer][idx]
            row_to_add.append(curr_activations)
        df.loc[len(df)] = row_to_add
    df.to_csv(filename)


def do_pca(X_in, n_components, show_plot=False):
    """Perform Principal Component Analysis on X_in data

    Args:
        X_in (array): input data
        n_components (int): number of components for PCA
        show_plot (bool, optional): Whether to show plot. Defaults to False.

    Returns:
        array: Principal Components of data
    """
    pca = PCA(n_components=n_components)
    X = pca.fit_transform(X_in)
    if show_plot:
        variances_explained = pca.explained_variance_ratio_
        cum_sum = np.cumsum(variances_explained)
        title = f"Cumulative Variance Explained by PC"
        fig = px.bar(x=range(1, len(cum_sum) + 1), y=cum_sum, title=title)
        fig.update_layout(title_font_size=24, font_size=20)
        fig.show()
    return X


def do_tsne(X_in, n_components, show_plot=False):
    tsne = TSNE(n_components=n_components, init="random")
    X = tsne.fit_transform(X_in)
    return X


def row_to_nparray(X):
    # convert a row from file generated by 'ActivationsToCSV' to an np array.
    return np.fromstring(X[1:-1], dtype=float, sep=" ")


def plot_accs(
    accuracies, words, X, title="", show_fig=True, save_fig=False, file_name=""
):
    df = pd.DataFrame(
        {
            "pca1": X[:, 0],
            "pca2": X[:, 1],
            "accuracies": accuracies,
            "words": words,
            # "predicted_class": labels,
        }
    )
    fig = px.scatter(
        df,
        x="pca1",
        y="pca2",
        color="accuracies",
        hover_data=["words"],
        title=title,
        height=500,
        width=800,
    )
    if save_fig:
        with open(file_name, "wb") as file:
            write_image(fig=fig, file=file, format="png")
    if show_fig:
        fig.show()


def plot_cv_scores(
    layer_scores,
    num_splits,
    num_layers,
    dict_key="layer_{}",
    save=True,
    show=True,
    save_name="img/cv_scores.png",
):
    """Given a dictionary, plots cross validated pearson coefficient for linear model trained on activations from layers of GPT2

    Input(s)
    -------
    layer_scores: dict - generated from 'train_regressors'
    dict_key: str - key used to index into 'layer_scores'
    save: bool - saves file with 'save_name'
    show: bool - shows plot
    """

    # collect the means (through this mean ahaha)
    means = []

    # scatter scores for each layer
    for n_layer in range(num_layers):
        # calculate score
        scores = layer_scores[dict_key.format(n_layer)]["cv_scores"]
        means.append(layer_scores[dict_key.format(n_layer)]["mean_score"])
        plt.scatter([n_layer] * len(scores), scores, s=5 ** 2, alpha=0.5)
        plt.scatter([n_layer], [np.mean(scores)], s=3 ** 2, marker="o", color="r")

    plt.plot([i for i in range(num_layers)], means)
    plt.xlabel("GPT2 Layer Number")
    plt.ylabel(f"Cross validation score ({num_splits} split(s))")

    if save:
        plt.rcParams["axes.facecolor"] = "white"
        plt.savefig(save_name, dpi=600)
    if show:
        plt.show()


def save_linear_model(model, filename):
    with open(filename, "wb") as f:
        pickle.dump(model, f)


def read_linear_model(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)

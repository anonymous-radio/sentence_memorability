"""
generate sentence embeddings for all sentences
different methods: sentence-bert, infersent, universal sentence encoder
glove mean embedding, gpt2, bert, resnet50
"""


from importlib_metadata import itertools
import pandas as pd
import numpy as np
import argparse
import embedding_utils as utils
from sklearn import linear_model
import transformers
import nlu
from sentence_transformers import SentenceTransformer
import tensorflow_hub as hub
import string
from tqdm import tqdm
from itertools import chain
from sklearn.metrics.pairwise import cosine_distances
import torch
import getpass
import typing
import os
from collections import defaultdict
from tqdm import tqdm
import re
import ast

torch.set_default_dtype(torch.double)


class ANNEncoder:
    _source_model = None

    def __init__(
        self,
        source_model: str = "gpt2",
        sent_embed: str = "last-tok",
        # these are used to define the cache, thus make them a part of the constructor
        actv_cache_setting: typing.Union[
            str, None
        ] = "auto",  # if auto, use cache. If None, don't use cache
        actv_cache_path: typing.Union[str, None] = None,
    ) -> None:

        self._source_model = source_model

        # Pretrained model
        from transformers import AutoModel, AutoConfig, AutoTokenizer

        self.config = AutoConfig.from_pretrained(self._source_model)
        self.tokenizer = AutoTokenizer.from_pretrained(self._source_model)
        self.model = AutoModel.from_pretrained(self._source_model, config=self.config)
        self.sent_embed = sent_embed

        # Cache
        self.user = getpass.getuser()
        self.actv_cache_setting = actv_cache_setting
        self.actv_cache_path = actv_cache_path

    # assert (self.config.num_hidden_layers == len(hidden_states))

    def _aggregate_layers(
        self, hidden_states: dict, sent_embed: str = "last-tok"
    ) -> None:
        """[summary]
        Args:
            hidden_states (torch.Tensor): pytorch tensor of shape (n_items, dims)
            sent_embed: an object specifying the method to use for aggregating
                                                    representations across items within a layer
        Raises:
            NotImplementedError
        Returns:
            np.ndarray: the aggregated array
        """
        states_layers = dict()
        for i in hidden_states.keys():  # for each layer
            if sent_embed == "last-tok":
                state = hidden_states[i][-1, :]  # get last token
            elif sent_embed == "first-tok":
                state = hidden_states[i][0, :]  # get first token
            elif sent_embed == "mean-tok":
                state = torch.mean(hidden_states[i], dim=0)  # mean over tokens
            elif sent_embed == "median-tok":
                state = torch.median(hidden_states[i], dim=0)  # median over tokens
            elif sent_embed == "sum-tok":
                state = torch.sum(hidden_states[i], dim=0)  # sum over tokens
            elif sent_embed == "all-tok" or sent_embed == None:
                state = hidden_states
            else:
                raise NotImplementedError("Sentence embedding method not implemented")

            states_layers[i] = state.detach().numpy()

        return states_layers

    def _flatten_activations(
        self, states_sentences_agg: dict, index: str = "DEFAULTINDEX"
    ):
        """Flatten activations.

        Args:
            states_sentences_agg (dict): dictionary of activations (key: layer, value: activations)
            index (str, optional): index to use for flattening (usually the stimid index). Defaults to 'DEFAULTINDEX'.

        Returns:
            df (pandas.DataFrame): flattened activations. Rows are sentences (indexed by index),
                columns are units flattened across layers.
        """

        labels = []
        lst_arr_flat = []
        for layer, arr in states_sentences_agg.items():
            arr = np.array(arr)  # for each layer
            lst_arr_flat.append(arr)
            # Create multiindex for each layer. index 0 is the layer index, and index 1 is the unit index
            for i in range(arr.shape[0]):  # across units
                labels.append((layer, i))
        arr_flat = np.concatenate(
            lst_arr_flat
        )  # concatenated activations across layers
        df = pd.DataFrame(arr_flat).T
        df.index = [index]
        df.columns = pd.MultiIndex.from_tuples(labels)  # rows: stimuli, columns: units
        return df

    def _create_actv_cache_path(self,):
        os.makedirs(self.actv_cache_path, exist_ok=True)

    def _case(self, sample: str = None, case: typing.Union[str, None] = None):
        if case == "lower":
            sample = sample.lower()
        elif case == "upper":
            sample = sample.upper()
        else:
            sample = sample

        return sample

    def get_special_token_offset(self) -> int:
        """
        the offset (no. of tokens in tokenized text) from the start to exclude
        when extracting the representation of a particular stimulus. this is
        needed when the stimulus is evaluated in a context group to achieve
        correct boundaries (otherwise we get off-by-context errors)
        """
        with_special_tokens = self.tokenizer("brainscore")["input_ids"]
        first_token_id, *_ = self.tokenizer("brainscore", add_special_tokens=False)[
            "input_ids"
        ]
        special_token_offset = with_special_tokens.index(first_token_id)
        return special_token_offset

    def get_context_groups(
        self, stimset: pd.DataFrame = None, context_dim: typing.Union[str, None] = None,
    ):
        """"Initialize the context group coordinate (obtain embeddings with context)"""
        if context_dim is None:
            context_groups = np.arange(0, len(stimset), 1)
        else:
            context_groups = stimset[context_dim].values

        return context_groups

    def encode(
        self,
        stimset: pd.DataFrame = None,
        stim_col: str = "sentence",
        case: typing.Union[str, None] = None,
        context_dim: str = None,
        bidirectional: bool = False,
        include_special_tokens: bool = True,
        cache_new_actv: bool = True,
        verbose: bool = False,
        **kwargs,
    ):
        """ Input a pandas dataframe with stimuli, encode and return a pandas dataframe with activations.

        Args:
            stimset (pd.DataFrame): a pandas dataframe with stimuli. If caching is enabled, the index has to adhere to 'stimid.0'
            stim_col (str): the column in the dataframe with the stimuli
            case (str): the case to use for the stimuli
            context_dim (str): the dimension to use for the context groupings of sampleids (stimuli) that should be used
                                                as context when generating encoder representations.
                                                If None, then no context is used.
            bidirectional (bool): if True, allows using "future" context to generate the representation for a current token
                                            otherwise, only uses what occurs in the "past". some might say, setting this to False
                                            gives you a more biologically plausibly analysis downstream.
        Raises:
            NotImplementedError: [description]
            ValueError: [description]
        Returns:
            [type]: [description]
        """
        # Obtain stimsetid (the identifier for the stimuli)
        stimsetid_all = [
            ".".join(stimset.index[x].split(".")[:-1]) for x in range(len(stimset))
        ]  # include all information separated by '.' besides the very last index
        assert (
            len(np.unique(stimsetid_all)) == 1
        )  # Check whether all sentences come from the same corpus
        stimsetid = stimsetid_all[0]
        if kwargs.get("stimsetid_suffix"):
            stimsetid_suffix = kwargs.get("stimsetid_suffix")
            # Add "_" to the end of the stimsetid if it doesn't already have one
            if stimsetid_suffix[0] != "_":
                stimsetid_suffix = "_" + stimsetid_suffix
            stimsetid = f"{stimsetid}{stimsetid_suffix}"

        self.stimset = stimset
        self.stimsetid = stimsetid
        self.stim_col = stim_col

        stim_fname = f"{self.stimsetid}_stim.pkl"
        actv_fname = f"{self.stimsetid}_actv.pkl"

        ### Check if we have already computed activations for this corpus (stimsetid) ###
        if self.actv_cache_setting == "auto":
            self._create_actv_cache_path()
            stim_fname = f"{self.actv_cache_path}/{stim_fname}"
            actv_fname = f"{self.actv_cache_path}/{actv_fname}"
            if os.path.exists(f"{actv_fname}"):
                print(
                    f"Loading cached ANN encoder activations for {self.stimsetid} from {self.actv_cache_path}\n"
                )
                stim = pd.read_pickle(stim_fname)
                actv = pd.read_pickle(actv_fname)
                assert (self.stimset.index == stim.index).all()
                assert (actv.index == stim.index).all()
                self.encoded_ann = actv
                return self.encoded_ann

        self.model.eval()
        stimuli = self.stimset[self.stim_col].values

        # Initialize the context group coordinate (obtain embeddings with context)
        context_groups = self.get_context_groups(
            stimset=stimset, context_dim=context_dim
        )

        ###############################################################################
        # ALL SAMPLES LOOP
        ###############################################################################
        states_sentences_across_groups = []
        stim_index_counter = 0
        _, unique_ixs = np.unique(context_groups, return_index=True)
        for group in tqdm(
            context_groups[np.sort(unique_ixs)]
        ):  # Make sure context group order is preserved
            mask_context = context_groups == group
            stim_in_context = stimuli[mask_context]  # Mask based on the context group

            states_sentences_across_stim = (
                []
            )  # Store states for each sample in this context group

            ###############################################################################
            # CONTEXT LOOP
            ###############################################################################
            for i, stimulus in enumerate(stim_in_context):
                stimulus = self._case(sample=stimulus, case=case)

                if len(stim_in_context) > 1:
                    print(f"encoding stimulus {i} of {len(stim_in_context)}")

                # mask based on the uni/bi-directional nature of models :)
                if not bidirectional:
                    stim_directional = stim_in_context[: i + 1]
                else:
                    stim_directional = stim_in_context

                # join the stimuli together within a context group using just a single space
                stim_directional = " ".join(stim_directional)

                stim_directional = self._case(sample=stim_directional, case=case)

                tokenized_directional_context = self.tokenizer(
                    stim_directional,
                    padding=False,
                    return_tensors="pt",
                    add_special_tokens=True,
                )

                # Get the hidden states
                result_model = self.model(
                    tokenized_directional_context.input_ids,
                    output_hidden_states=True,
                    return_dict=True,
                )
                hidden_states = result_model[
                    "hidden_states"
                ]  # dict with key=layer, value=3D tensor of dims: [batch, tokens, emb size]

                layerwise_activations = defaultdict(list)

                # Find which indices match the current stimulus in the given context group
                start_of_interest = stim_directional.find(stimulus)
                char_span_of_interest = slice(
                    start_of_interest, start_of_interest + len(stimulus)
                )
                token_span_of_interest = pick_matching_token_ixs(
                    tokenized_directional_context, char_span_of_interest
                )

                if verbose:
                    print(
                        f"Interested in the following stimulus:\n{stim_directional[char_span_of_interest]}\n"
                        f"Recovered:\n{tokenized_directional_context.tokens()[token_span_of_interest]}"
                    )  # See which tokens are used (with the special tokens)

                all_special_ids = set(self.tokenizer.all_special_ids)

                # Look for special tokens in the beginning and end of the sequence
                insert_first_upto = 0
                insert_last_from = tokenized_directional_context.input_ids.shape[-1]
                # loop through input ids
                for i, tid in enumerate(tokenized_directional_context.input_ids[0, :]):
                    if tid.item() in all_special_ids:
                        insert_first_upto = i + 1
                    else:
                        break
                for i in range(
                    1, tokenized_directional_context.input_ids.shape[-1] + 1
                ):
                    tid = tokenized_directional_context.input_ids[0, -i]
                    if tid.item() in all_special_ids:
                        insert_last_from -= 1
                    else:
                        break

                for idx_layer, layer in enumerate(hidden_states):  # Iterate over layers
                    # b (1), n (tokens), h (768, ...)
                    # collapse batch dim to obtain shape (n_tokens, emb_dim)
                    this_extracted = layer[:, token_span_of_interest, :,].squeeze(0)

                    if include_special_tokens:
                        # get the embeddings for the first special tokens
                        this_extracted = torch.cat(
                            [
                                layer[:, :insert_first_upto, :].squeeze(0),
                                this_extracted,
                            ],
                            axis=0,
                        )
                        # get the embeddings for the last special tokens
                        this_extracted = torch.cat(
                            [
                                this_extracted,
                                layer[:, insert_last_from:, :].squeeze(0),
                            ],
                            axis=0,
                        )

                    layerwise_activations[idx_layer] = this_extracted.detach()

                # aggregate within a stimulus
                states_sentences_agg = self._aggregate_layers(
                    layerwise_activations, sent_embed=self.sent_embed
                )
                # dict with key=layer, value=array of # size [emb dim]

                # Convert to flattened pandas df
                current_stimid = stimset.index[stim_index_counter]
                assert (
                    self._case(sample=stimset.loc[current_stimid][stim_col], case=case)
                    == stimulus
                )

                df_states_sentences_agg = self._flatten_activations(
                    states_sentences_agg, index=current_stimid
                )

                # append the dfs to states_sentences_across_stim (which is ALL stim within a context group)
                states_sentences_across_stim.append(df_states_sentences_agg)
                # now we have all the hidden states for the current context group across all stimuli

                stim_index_counter += 1

            ###############################################################################
            # END CONTEXT LOOP
            ###############################################################################

            states_sentences_across_groups.append(
                pd.concat(states_sentences_across_stim, axis=0)
            )

        ###############################################################################
        # END ALL SAMPLES LOOP
        ###############################################################################

        actv = pd.concat(states_sentences_across_groups, axis=0)

        print(
            f"Number of stimuli in activations: {actv.shape[0]}\n"
            f"Number of units in activations: {actv.shape[1]}\n"
        )

        assert (stimset.index == actv.index).all()

        if cache_new_actv:
            stimset.to_pickle(stim_fname, protocol=4)
            actv.to_pickle(actv_fname, protocol=4)
            print(
                f"\nCaching newly computed activations!\nCached activations to {actv_fname}"
            )

        self.encoded_ann = actv
        self.encoded_stimset = stimset

        return self.encoded_ann


### ENCODER UTILS FUNCTIONS ###


def pick_matching_token_ixs(
    batchencoding: "transformers.tokenization_utils_base.BatchEncoding",
    char_span_of_interest: slice,
    verbose: bool = False,
) -> slice:
    """Picks token indices in a tokenized encoded sequence that best correspond to
        a substring of interest in the original sequence, given by a char span (slice)

    Args:
        batchencoding (transformers.tokenization_utils_base.BatchEncoding): the output of a
            `tokenizer(text)` call on a single text instance (not a batch, i.e. `tokenizer([text])`).
        char_span_of_interest (slice): a `slice` object denoting the character indices in the
            original `text` string we want to extract the corresponding tokens for

    Returns:
        slice: the start and stop indices within an encoded sequence that
            best match the `char_span_of_interest`
    """
    from transformers import tokenization_utils_base

    start_token = 0
    end_token = batchencoding.input_ids.shape[-1]
    for i, _ in enumerate(batchencoding.input_ids.reshape(-1)):
        span = batchencoding[0].token_to_chars(
            i
        )  # batchencoding 0 gives access to the encoded string

        if span is None:  # for [CLS], no span is returned
            if verbose:
                print(
                    f'No span returned for token at {i}: "{batchencoding.tokens()[i]}"'
                )
            continue
        else:
            span = tokenization_utils_base.CharSpan(*span)

        if span.start <= char_span_of_interest.start:
            start_token = i
        if span.end >= char_span_of_interest.stop:
            end_token = i + 1
            break

    assert (
        end_token - start_token <= batchencoding.input_ids.shape[-1]
    ), f"Extracted span is larger than original span"

    return slice(start_token, end_token)


def strip_punc_sent(s):
    return " ".join([w.strip(string.punctuation) for w in s.lower().split()])


def get_embeddings_sbert(sentences):
    embeddings = []
    model = SentenceTransformer("bert-base-nli-mean-tokens")
    for split in tqdm(np.array_split(sentences, 20)):
        embeddings.append(model.encode(split))
    return np.array(list(chain.from_iterable(embeddings)))


def get_embeddings_use(sentences):
    embeddings = []
    model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
    for split in tqdm(np.array_split(sentences, 20)):
        embeddings.append(model(split))
    return np.array(list(chain.from_iterable(embeddings)))


def get_embeddings_electra(sentences):
    embeddings = []
    pipe = nlu.load("embed_sentence.electra")
    for split in tqdm(np.array_split(sentences, 20)):
        print("Length of split:", len(split))
        x = pipe.predict(split, output_level="document").sentence_embedding_electra.tolist()
        print(len(x))
        embeddings.append(x)
    return np.array(list(chain.from_iterable(embeddings)))


def get_embeddings_mean_glove(sentences):
    sentences = list(map(strip_punc_sent, sentences))
    words = list(set(itertools.chain.from_iterable([x.split() for x in sentences])))
    embs, oovs = utils.get_embeddings_glove(words, embs_file=args.glove_file)
    mapping = dict(zip(words, embs))
    dimsize = embs[0].shape[0]
    for oov in oovs:
        mapping[oov] = np.full(dimsize, np.nan)
    return np.array([np.nanmean([mapping.get(word) for word in sentence.split()], axis=0) for sentence in sentences])


def get_embeddings_infersent(sentences):
    pass


def run_regression(args, embeddings, accuracies):
    """run Cross-Validation regression on sentence embeddings and accuracies

    Args:
        args (_type_): _description_
        embeddings (np.array): array of embeddings
        accuracies (np.array): array of accuracies

    Returns:
        sklearn.LinearModel: linear regression model
    """
    regr = linear_model.LinearRegression()
    scores, _, _ = utils.cross_validation(
        embeddings, accuracies, regr, k=args.num_splits, group_values=embeddings
    )
    print("Cross-Validated Regression Scores:", scores)
    print("Mean Score:", np.mean(scores))
    utils.save_linear_model(regr, f"linear_models_sent/{args.embedding_method}.pkl")
    return regr


def distinctiveness(args, embeddings):
    """Return the mean cosine distance of each embedding to every other embedding

    Args:
        args (_type_): _description_
        embeddings (np.array): array of embeddings

    Returns:
        np.array: array of mean cosine distances
    """
    dists = cosine_distances(embeddings, embeddings)
    for i in range(dists.shape[0]):
        dists[i, i] = np.nan
    return np.nanmean(dists, axis=1).flatten()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_splits", type=int, default=10)
    parser.add_argument(
        "--csv", default="../analysis/materials_w-sent-identifier.csv")
    parser.add_argument("--embedding_model", default="gpt2-medium")
    parser.add_argument("--embedding_model_layer", type=int, default=12)
    parser.add_argument("--num_layers", type=int, default=24)
    parser.add_argument("--sent_embed", default="last-tok")
    parser.add_argument("--regressions", action="store_true")
    parser.add_argument("--distinctiveness", action="store_true")
    parser.add_argument("--glove_file", default="glove.42B.300d.txt")
    args = parser.parse_args()

    # load sentences
    df_stimset = pd.read_csv(args.csv)

    # Add a meaningful index: sentmem.{item_id}
    df_stimset["sentmem_id"] = df_stimset.apply(lambda row: f"sentmem.{row['item_id']}", axis=1)
    df_stimset = df_stimset.set_index("sentmem_id").sort_values(by='item_id')

    if not (args.embedding_model.startswith("gpt") or args.embedding_model.startswith("bert")):
        df_stimset = df_stimset[df_stimset.condition.isin(["high", "mid", "low"])]
    else: 
        df_stimset = df_stimset[df_stimset.condition.isin(["misc", "high", "mid", "low"])]

    print(df_stimset)

    if args.embedding_model == "use":
        embeddings = get_embeddings_use(df_stimset.sentence.tolist())
    elif args.embedding_model == "infersent":
        embeddings = get_embeddings_infersent(df_stimset.sentence.tolist())
    elif args.embedding_model == "sbert":
        embeddings = get_embeddings_sbert(df_stimset.sentence.tolist())
    elif args.embedding_model == "electra":
        embeddings = get_embeddings_electra(df_stimset.sentence.tolist())
    elif args.embedding_model == "glove":
        embeddings = get_embeddings_mean_glove(df_stimset.sentence.tolist())
    elif args.embedding_model == "resnet50":
        df = pd.read_csv("../model-actv/resnet50/output.csv").sort_values("item_id")
        # print(df)
        embeddings = df.resnet50_output.apply(lambda x: re.sub("\s+", ",", x).replace(",]", "]").replace("[,", "["))
        # print(embeddings[0])
        embeddings = np.array([np.array(ast.literal_eval(x)) for x in list(embeddings)])
    elif args.embedding_model == "resnet50-early":
        df = pd.read_pickle("../model-actv/resnet50-early/act1_actv_no-misc.pickle")# .sort_values("item_id")
        # print(df)
        embeddings = df.to_numpy()
    elif args.embedding_model.startswith("gpt") or args.embedding_model.startswith("bert"):

        ####### ANN ENCODER ########
        ann = ANNEncoder(source_model=args.embedding_model,
                         sent_embed=args.sent_embed,
                         actv_cache_setting='auto',
                         actv_cache_path=f'../model-actv/{args.embedding_model}/{args.sent_embed}/')

        embeddings = ann.encode(stimset=df_stimset,
                   cache_new_actv=True,
                   case=None,
                   **{'stimsetid_suffix': f''},
                   include_special_tokens=True,
                   verbose=False)

        # hacky: this is to exclude the misc sentences (first 1000)
        embeddings = embeddings.iloc[1000:, :] 


    else:
        raise ValueError(f"Unrecognized embedding model: {args.embedding_model}")

    print("Embedding shape:", embeddings.shape)
    print("Accuracy shape:", df_stimset.correct.shape)

    if args.regressions:
        regr = run_regression(args, embeddings, df_stimset.correct)
    if args.distinctiveness:
        if args.embedding_model.startswith("gpt") or args.embedding_model.startswith("bert"):
            df_stimset = df_stimset[df_stimset.condition.isin(["high", "mid", "low"])]
            for model_layer in range(args.num_layers+1):
                avg_dists = distinctiveness(args, embeddings[model_layer])
                df_stimset[f"avg_dist_{args.embedding_model}" + f"_layer-{model_layer}"] = avg_dists

        else:
            avg_dists = distinctiveness(args, embeddings)
            df_stimset[f"avg_dist_{args.embedding_model}"] = avg_dists

        df_stimset.to_csv(f"critical_acc_v_avg_dist_{args.embedding_model}.csv", index=False)

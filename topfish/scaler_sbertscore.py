# import cupy as np
import numpy as np
import numba as nb
import torch
from numba.typed import List
from helpers import io_helper
from graphs import graph
from datetime import datetime
import time
import pickle
from sts import simple_sts
from sentence_transformers import SentenceTransformer, util
from pathlib import Path
from timer import timer
from nlp import common_component_removal
from tqdm import tqdm
from spacy.lang.xx import MultiLanguage
from collections import Counter
MAXIMUM_N_SENTENCES = None  # If Float, e.g. 2.0 = every second sentence.
MAXIMUM_WORDS = None

SPACE_LANGUAGE_MAPPING = {"multi" : MultiLanguage()}

PATH_SAVE_EMBEDDINGS = Path(__file__).parents[1] / "embeddings" / "sbert"
MODELNAME = "paraphrase-multilingual-mpnet-base-v2"
# MODELNAME = "all-mpnet-base-v2"
# MODELNAME = "multi-qa-mpnet-base-dot-v1"
BATCH_SIZE = 16


def scale_sbertscore(filenames, texts, languages, predictions_file_path):
    """Scaling with SBERT embeddingds and BERTScore.

    Args:
        filenames: list of filenames, e.g. ['input_16.txt', 'input_49.txt']
        texts: list of texts, e.g. ["this is the text of input_16", "this is the text of input_49"]
        languages: list of language per text, e.g. ["english", "english"]
        predictions_file_path: Text file to save the scaling results, e.g. output_scaler/scale_sbertscore/english/eu/8/english_eu_8.txt

    Returns:

    """
    #languages = ['english', 'irish']
    # Create/Get SBERT Embeddings
    nested_sentences_embeddings = get_sbert_embeddings(texts, predictions_file_path, languages)

    # Do Common Component Removal
    nested_sentences_embeddings = embedding_common_component_removal(nested_sentences_embeddings)

    print(datetime.now().strftime('%Y-%m-%d %H:%M:%S') + " Building BERTScore.", flush=True)

    # Bert Score
    sbert_pairs_file = Path(predictions_file_path).parents[0] / "sbert_pairs" / 'sbert_pairs.pkl'
    if sbert_pairs_file.is_file():
        print(datetime.now().strftime('%Y-%m-%d %H:%M:%S') + " Load Saved BertScores.", flush=True)
        with open(sbert_pairs_file, "rb") as fIn:
            pairs = pickle.load(fIn)
    else:
        Path(sbert_pairs_file.parents[0]).mkdir(parents=True, exist_ok=True)
        #pairs = sbert_cos_similarity(nested_sentences_embeddings, filenames)
        #pairs = bert_score(filenames, nested_sentences_embeddings)
        pairs = combine_bertscore_cos(filenames, nested_sentences_embeddings)

    with open(sbert_pairs_file, "wb") as fOut:
        pickle.dump(pairs, fOut, protocol=pickle.HIGHEST_PROTOCOL)

    # rescale distances and produce similarities
    print(datetime.now().strftime('%Y-%m-%d %H:%M:%S') + " Normalizing pairwise similarities.", flush=True)
    max_sim = max([x[2] for x in pairs])
    min_sim = min([x[2] for x in pairs])
    pairs = [(x[0], x[1], (x[2] - min_sim) / (max_sim - min_sim)) for x in pairs]

    print(datetime.now().strftime('%Y-%m-%d %H:%M:%S') + " Fixing the pivot documents for scaling.", flush=True)
    min_sim_pair = [x for x in pairs if x[2] == 0][0]
    fixed = [(filenames.index(min_sim_pair[0]), -1.0), (filenames.index(min_sim_pair[1]), 1.0)]

    # propagating position scores, i.e., scaling
    print(datetime.now().strftime(
        '%Y-%m-%d %H:%M:%S') + " Running graph-based label propagation with pivot rescaling and score normalization.",
          flush=True)

     
    print([item for item, count in Counter(filenames).items() if count > 1])
    filenames = list(dict.fromkeys(filenames))
    g = graph.Graph(nodes=filenames, edges=pairs)
    
    print([item for item, count in Counter(g.nodes).items() if count > 1])
    scores = g.harmonic_function_label_propagation(fixed, rescale_extremes=True, normalize=True)

    if predictions_file_path:
        io_helper.write_dictionary(predictions_file_path, scores)

    return scores


def scale_sbertscore_supervised(filenames, texts, languages, predictions_file_path, pivot1, pivot2, stopwords=[],
                                emb_lang='default'):
    # Create/Get SBERT Embeddings
    nested_sentences_embeddings = get_sbert_embeddings(texts, predictions_file_path, languages)

    # Do Common Component Removal
    nested_sentences_embeddings = embedding_common_component_removal(nested_sentences_embeddings)
    print(datetime.now().strftime('%Y-%m-%d %H:%M:%S') + " Building BERTScore.", flush=True)
    # Bert Score
    sbert_pairs_file = Path(predictions_file_path).parents[0] / "sbert_pairs" / 'sbert_pairs.pkl'
    if sbert_pairs_file.is_file():
        with open(sbert_pairs_file, "rb") as fIn:
            pairs = pickle.load(fIn)
    else:
        Path(sbert_pairs_file.parents[0]).mkdir(parents=True, exist_ok=True)
        #pairs = sbert_cos_similarity(nested_sentences_embeddings, filenames)
        #pairs = bert_score(filenames, nested_sentences_embeddings)
        pairs = combine_bertscore_cos(filenames, nested_sentences_embeddings)
    with open(sbert_pairs_file, "wb") as fOut:
        pickle.dump(pairs, fOut, protocol=pickle.HIGHEST_PROTOCOL)

    # rescale distances and produce similarities
    print(datetime.now().strftime('%Y-%m-%d %H:%M:%S') + " Normalizing pairwise similarities...", flush=True)
    max_sim = max([x[2] for x in pairs])
    min_sim = min([x[2] for x in pairs])
    pairs = [(x[0], x[1], (x[2] - min_sim) / (max_sim - min_sim)) for x in pairs]

    print(datetime.now().strftime('%Y-%m-%d %H:%M:%S') + " Fixing the pivot documents for scaling...", flush=True)
    min_sim_pair = [pivot1, pivot2, 0.0]

    fixed = [(filenames.index(min_sim_pair[0]), -1.0), (filenames.index(min_sim_pair[1]), 1.0)]
    #       fixed = [(pivot1, -1.0), (pivot2, 1.0)]

    # propagating position scores, i.e., scaling
    print(datetime.now().strftime(
        '%Y-%m-%d %H:%M:%S') + " Running graph-based label propagation with pivot rescaling and score normalization...",
          flush=True)
    g = graph.Graph(nodes=filenames, edges=pairs)
    scores = g.harmonic_function_label_propagation(fixed, rescale_extremes=False, normalize=True)

    if predictions_file_path:
        io_helper.write_dictionary(predictions_file_path, scores)
    return scores


# Print iterations progress
def printProgressBar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█', printEnd="\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()


@timer
def recreate_nested_list(flattened_list, len_items_of_nested_list):
    """Recreated the nested structure before flattening the list. Useful to get the corresponding sentences of each
    paragraph back after flattening the paragraphs.

    Args:
        flattened_list:
        len_items_of_nested_list:

    Returns:

    """
    if isinstance(flattened_list, np.ndarray):
        flattened_list = flattened_list.tolist()

    def nested(vals, start, end):
        vals[start:end] = np.array([vals[start:end]])
        return vals

    end = len(flattened_list)
    for length_item in list(reversed(len_items_of_nested_list)):
        flattened_list = nested(flattened_list, end - length_item, end)
        end = end - length_item

    return flattened_list


@timer
def flatten_list(nested_list):
    """Flatten a nested list.

    Args:
        nested_list:

    Returns:

    """
    return [item for single_list in nested_list for item in single_list]


@timer
def spacy_split_into_sentences(texts, languages):
    """Splits text into sentences using spacy.

    Args:
        texts:
        languages:

    Returns:
         sentences_texts is a list of list, each element (list) corresponds to a party speech, splitted into sentences

    """
    nlp = SPACE_LANGUAGE_MAPPING[languages[0]]
    try:
        nlp.add_pipe('sentencizer')
        nlp.max_length = 20000000
    except ValueError:
        pass
    sentence_texts = []
    for text in texts:
        if MAXIMUM_WORDS:
            text = text[:MAXIMUM_WORDS]
        doc = nlp(text)
        new_text = [sent.text.strip() for sent in doc.sents]
        if MAXIMUM_N_SENTENCES:
            if isinstance(MAXIMUM_N_SENTENCES, int):
                new_text = new_text[:MAXIMUM_N_SENTENCES]
            elif isinstance(MAXIMUM_N_SENTENCES, float):
                new_text = new_text[::int(MAXIMUM_N_SENTENCES)]
        sentence_texts.append(new_text)
    return sentence_texts


@timer
def rulebased_split_into_sentences(texts, languages):
    sentence_texts = []
    for text in texts:
        if MAXIMUM_WORDS:
            text = text[:MAXIMUM_WORDS]
        text_split = text.split(".")
        new_text = [text_with_spaces.strip() + "." for text_with_spaces in text_split if len(text_with_spaces) != 0]
        if MAXIMUM_N_SENTENCES:
            new_text = new_text[:MAXIMUM_N_SENTENCES]
        sentence_texts.append(new_text)
    return sentence_texts


def sbert_cos_similarity(emb_sentences, filenames):
    mean_emb = np.array([np.mean(x, axis=0) for x in emb_sentences])
    pairs = simple_sts.fast_cosine_similarity(mean_emb, filenames)
    return pairs


def save_sbert_embeddings(sentences, embeddings, file):
    with open(file, "wb") as fOut:
        pickle.dump({'sentences': sentences, 'embeddings': embeddings},
                    fOut, protocol=pickle.HIGHEST_PROTOCOL)


def load_sbert_embeddings(file, current_text=None):
    """Load SBERT Embeddings if exists. Automatically does sanity check if text correspond to the embeddings that are
    being loaded.

    Args:
        file:
        current_text:

    Returns:

    """
    with open(file, "rb") as fIn:
        loaded_data = pickle.load(fIn)
        nested_sentences_embeddings = loaded_data['embeddings']
        loaded_text = loaded_data['sentences']
        if current_text:
            # Sanity Check
            assert loaded_text == current_text, "Saved Embeddings do not correspond to the input text. " \
                                                "Please delete current embeddings in {} to create new embeddings " \
                                                "for your input text".format(file)
    return nested_sentences_embeddings


def normalized(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)


def embedding_common_component_removal(nested_sentences_embeddings):
    """Common Component Removal for list of np.arrays.

    Args:
        nested_sentences_embeddings:

    Returns:

    """
    len_sentences = [len(sentences) for sentences in nested_sentences_embeddings]

    # Flatten list
    flatten_sentences_texts = flatten_list(nested_sentences_embeddings)

    # Stack each sentence embedding
    stacked_embeddings = np.vstack(flatten_sentences_texts)
    stacked_embeddings = common_component_removal(stacked_embeddings, pc=1)

    stacked_embeddings = normalized(stacked_embeddings, axis=1, order=2)

    # Recreated the nested structure before flattening the list.
    nested_sentences_embeddings = recreate_nested_list(stacked_embeddings, len_sentences)
    return nested_sentences_embeddings


@timer
def get_sbert_embeddings(texts, predictions_file_path, languages):
    """Creates or loads (if exists) SBERT Embeddings.

    Args:
        texts:
        predictions_file_path:
        languages:

    Returns:

    """
    # embedding file path
    embedding_file = Path(predictions_file_path).parents[0] / "embedding" / 'embeddings.pkl'

    # Break up paragraphs into sentences. Feed them separately into the model.
    # sentences_texts is a list of list, each element (list) corresponds to a party speech, splitted into sentences
    sentences_texts = spacy_split_into_sentences(texts, languages)

    # Debug: Some Key Information about the process, e.g. Number of cosine similarity operations
    len_sentences = [len(sentences) for sentences in sentences_texts]
    n_words = []
    for sentences in sentences_texts:
        text_n_words = 0
        for sentence in sentences:
            text_n_words += len(sentence)
        n_words.append(text_n_words)
    print("Number of words: {}".format(n_words))
    n_cos_operations = 0
    for i in range(len(sentences_texts) - 1):
        for j in range(i + 1, len(sentences_texts)):
            n_cos_operations += len(sentences_texts[i]) * len(sentences_texts[j])
    print("Number of cosine similarity operations: {}".format(str(n_cos_operations)))
    n_embedding_operations = 0
    for sentence_text in sentences_texts:
        n_embedding_operations = n_embedding_operations + len(sentence_text)

    print("Number of embedding operations: {}".format(str(n_embedding_operations)))

    # Flatten List, so we can utilize batches in transformers
    flatten_sentences_texts = flatten_list(sentences_texts)

    if embedding_file.is_file():
        # Load Embeddings if embedding file already exist. Does sanity check if embeddings correspond to current text.
        print(datetime.now().strftime('%Y-%m-%d %H:%M:%S') + " Embeddings already exits. "
                                                             "Load existing embeddings.", flush=True)
        nested_sentences_embeddings = load_sbert_embeddings(embedding_file, current_text=sentences_texts)

    else:
        # Create SBERT Embeddings by loading a pretrained model.
        print(datetime.now().strftime('%Y-%m-%d %H:%M:%S') + " Create SBERT.", flush=True)
        # input: texts: contains all text (n_paragraphs, text); filenames: contains corresponding names
        model = SentenceTransformer(MODELNAME, device='cuda')

        Path(embedding_file.parents[0]).mkdir(parents=True, exist_ok=True)
        s = time.time()
        print(datetime.now().strftime('%Y-%m-%d %H:%M:%S') + " Create embeddings with SBERT.", flush=True)
        sentence_embeddings = model.encode(flatten_sentences_texts,
                                           batch_size=BATCH_SIZE,
                                           show_progress_bar=True,
                                           convert_to_numpy=True,
                                           normalize_embeddings=True)

        print(datetime.now().strftime('%Y-%m-%d %H:%M:%S') +
              " Finished Embedding in {}s.".format(time.time() - s), flush=True)

        # Recreated the nested structure before flattening the list.
        nested_sentences_embeddings = recreate_nested_list(sentence_embeddings, len_sentences)

        print(datetime.now().strftime('%Y-%m-%d %H:%M:%S') + " Saving Embeddings.", flush=True)
        save_sbert_embeddings(sentences_texts, nested_sentences_embeddings, embedding_file)

    return nested_sentences_embeddings


@timer
def cosine_similarity(x, y):
    return x @ y.T


# https://github.com/numba/numba/issues/1269
@nb.njit
def np_apply_along_axis(func1d, axis, arr):
    assert arr.ndim == 2
    assert axis in [0, 1]
    if axis == 0:
        result = np.empty(arr.shape[1])
        for i in range(len(result)):
            result[i] = func1d(arr[:, i])
    else:
        result = np.empty(arr.shape[0])
        for i in range(len(result)):
            result[i] = func1d(arr[i, :])
    return result


@nb.njit
def np_amax(array, axis):
    """Custom np.amax function with numba support.

    Args:
        array:
        axis:

    Returns:

    """
    return np_apply_along_axis(np.amax, axis, array)


@timer
def bert_score(filenames, nested_sentences_embeddings):
    """

    Args:
        filenames: list, filenames
        nested_sentences_embeddings: list of np.array, each element is a political speeched embedded
                                     by SBERT (n_sentence, embedding_size).

    Returns:

    """

    # input: all paragraphs, each paragaph is (n_sentence, embedding_size).
    # ouput: triangle matrix, one coloumn = row = BERTScore for (Policy Area, Betrachtungszeitraum, Politiker)

    @nb.njit()
    def bertscore_fill_triangular_matrix(paragraphs):
        """Fills upper triangular matrix with bertscores

        Args:
            paragraphs: list of np.array, each element is a political speeched embedded
                        by SBERT (n_sentence, embedding_size).

        Returns:

        """
        print("\nStart Cosine Similarity Calculation!")
        N = len(paragraphs)
        M = np.zeros((N, N))
        for i in nb.prange(N - 1):
            # i = reference
            print("Cosine Calculation for Document: " + str(i) + "/" + str(N - 1))
            # TODO: Stack Vectors together for faster comutation?
            for j in nb.prange(i + 1, N):
                # j = candidate
                cos_sim_matrix = paragraphs[i] @ paragraphs[j].T

                # Normalize cosine similarities to [0, 1]
                cos_sim_matrix += 1
                cos_sim_matrix /= 2

                # Costume np.amax function since numba does not support axis argument
                amax_0 = np_amax(cos_sim_matrix, axis=0) / paragraphs[j].shape[0]
                amax_1 = np_amax(cos_sim_matrix, axis=1) / paragraphs[i].shape[0]
                P = np.sum(amax_0)
                R = np.sum(amax_1)
                F = 2 * P * R / (P + R)
                M[i, j] = F
        return M

    # nb.njit can not work with standard python list, have to convert it
    typed_sentences_embeddings = List()
    [typed_sentences_embeddings.append(x) for x in nested_sentences_embeddings]

    # Create BERTScore triangular matrix Score.
    bert_scores = bertscore_fill_triangular_matrix(typed_sentences_embeddings)

    # Build pairs from matrix [doc_x idx, doc_y idx , score]
    N_c = bert_scores.shape[0]
    pairs = []
    for i in range(N_c):
        for j in range(i + 1, N_c):
            pairs.append((filenames[i], filenames[j], bert_scores[i, j]))

    return pairs


def bert_score_gpu(filenames, nested_sentences_embeddings):
    """

        Args:
            filenames: list, filenames
            nested_sentences_embeddings: list of np.array, each element is a political speeched embedded
                                         by SBERT (n_sentence, embedding_size).

        Returns:

        """

    # input: all paragraphs, each paragaph is (n_sentence, embedding_size).
    # ouput: triangle matrix, one coloumn = row = BERTScore for (Policy Area, Betrachtungszeitraum, Politiker)
    def bertscore_fill_triangular_matrix(paragraphs):
        """Fills upper triangular matrix with bertscores

        Args:
            paragraphs: list of np.array, each element is a political speeched embedded
                        by SBERT (n_sentence, embedding_size).

        Returns:

        """
        print("\nStart Cosine Similarity Calculation!")
        N = len(paragraphs)
        M = np.zeros((N, N))
        for i in range(N - 1):
            # i = reference
            print("Cosine Calculation for Document: " + str(i) + "/" + str(N - 1))
            # TODO: Stack Vectors together for faster comutation?
            for j in range(i + 1, N):
                # j = candidate
                cos_sim_matrix = np.matmul(paragraphs[i], paragraphs[j].T)
                # Normalize cosine similarities to [0, 1]
                cos_sim_matrix += 1
                cos_sim_matrix /= 2

                # Costume np.amax function since numba does not support axis argument
                amax_0 = np.amax(cos_sim_matrix, axis=0) / paragraphs[j].shape[0]
                amax_1 = np.amax(cos_sim_matrix, axis=1) / paragraphs[i].shape[0]
                P = np.sum(amax_0)
                R = np.sum(amax_1)
                F = 2 * P * R / (P + R)
                M[i, j] = F

        return M

    # Create BERTScore triangular matrix Score.
    bert_scores = bertscore_fill_triangular_matrix(nested_sentences_embeddings)

    # Build pairs from matrix [doc_x idx, doc_y idx , score]
    N_c = bert_scores.shape[0]
    pairs = []
    for i in range(N_c):
        for j in range(i + 1, N_c):
            pairs.append((filenames[i], filenames[j], bert_scores[i, j]))

    return pairs


def normalize_list(list_normal):
    max_value = max(list_normal)
    min_value = min(list_normal)
    for i in range(len(list_normal)):
        list_normal[i] = (list_normal[i] - min_value) / (max_value - min_value)
    return list_normal


def greedy_cos_idf(ref_embedding, hyp_embedding):
    ref_embedding = torch.unsqueeze(ref_embedding, 0)
    hyp_embedding = torch.unsqueeze(hyp_embedding, 0)
    # ref_embedding.div_(torch.norm(ref_embedding, dim=-1).unsqueeze(-1))
    # hyp_embedding.div_(torch.norm(hyp_embedding, dim=-1).unsqueeze(-1))
    batch_size = ref_embedding.size(0)
    sim = torch.bmm(hyp_embedding, ref_embedding.transpose(1, 2))
    word_precision = sim.max(dim=2)[0]
    word_recall = sim.max(dim=1)[0]
    hyp_idf = torch.full(hyp_embedding.size()[:2], 1.0)
    ref_idf = torch.full(ref_embedding.size()[:2], 1.0)
    hyp_idf.div_(hyp_idf.sum(dim=1, keepdim=True))
    ref_idf.div_(ref_idf.sum(dim=1, keepdim=True))
    precision_scale = hyp_idf.to(word_precision.device)
    recall_scale = ref_idf.to(word_recall.device)

    P = (word_precision * precision_scale).sum(dim=1)
    R = (word_recall * recall_scale).sum(dim=1)
    F = 2 * P * R / (P + R)

    F = F.masked_fill(torch.isnan(F), 0.0)

    return P, R, F


def bertscore_pytorch(filenames, nested_sentences_embeddings):
    N = len(nested_sentences_embeddings)
    M = np.zeros((N, N))
    for i in range(N - 1):
        print("Cosine Calculation for Document: " + str(i) + "/" + str(N - 1))
        torch_embedding = torch.from_numpy(nested_sentences_embeddings[i])
        for j in range(i + 1, N):
            P, R, F = greedy_cos_idf(torch_embedding, torch.from_numpy(nested_sentences_embeddings[j]))
            M[i, j] = F
    N_c = M.shape[0]
    pairs = []
    for i in range(N_c):
        for j in range(i + 1, N_c):
            pairs.append((filenames[i], filenames[j], M[i, j]))

    return pairs


def combine_bertscore_cos(filenames, nested_sentences_embeddings):
    pairs_cos = sbert_cos_similarity(nested_sentences_embeddings, filenames)
    pairs_bertscore = bert_score(filenames, nested_sentences_embeddings)
    pairs_cos_values = normalize_list([x[2] for x in pairs_cos])
    pairs_bertscore_values = normalize_list([x[2] for x in pairs_bertscore])

    final_pairs = []
    for index in range(len(pairs_cos)):
        avg_score = (pairs_cos_values[index] + pairs_bertscore_values[index]) / 2
        if pairs_cos[index][0] != pairs_bertscore[index][0] or pairs_cos[index][1] != pairs_bertscore[index][1]:
            sys.exit("No same filename")
        final_pairs.append((pairs_cos[index][0], pairs_cos[index][1], avg_score))
    return final_pairs

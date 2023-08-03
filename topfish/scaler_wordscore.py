from nltk.tokenize import word_tokenize
from sts import simple_sts
from datetime import datetime
import pandas as pd
import numpy as np
from helpers import io_helper


def wordscore(filenames, texts, languages, predictions_file_path, pivot1, pivot2,
              pivot1_value, pivot2_value, stopwords):

    # Reference Texts
    Ar = pd.DataFrame({pivot1: -1, pivot2: 1}, index=['score'])

    # Tokenize Set
    print(datetime.now().strftime('%Y-%m-%d %H:%M:%S') + " Tokenizing documents.", flush=True)
    texts_tokenized = []
    for i in range(len(texts)):
        # print("Document " + str(i + 1) + " of " + str(len(texts)), flush = True)
        texts_tokenized.append(simple_sts.simple_tokenize(texts[i], stopwords, lang_prefix=None))

    doc_dicts = []
    cntr = 0
    print(datetime.now().strftime('%Y-%m-%d %H:%M:%S') + "  Building vocabularies for documents...", flush=True)
    for x in texts_tokenized:
        cntr += 1
        print("Document " + str(cntr) + " of " + str(len(texts)))
        doc_dicts.append(build_vocab(x, count_treshold=1))

    # load reference data
    Fwr = create_csv(Ar.keys(), filenames, doc_dicts, freq="rel")

    # compute p(r|w) = f_wr / sum(f_wr)_{for all r}
    Pwr = Fwr.iloc[:, 1:].div(Fwr.sum(axis=1), axis=0)

    # compute Sw and save to file Sw = sum_r(P_wr * A_rd)
    Sw = pd.DataFrame(Fwr.word)
    Sw['score'] = Pwr.to_numpy() @ Ar.to_numpy().T

    # Filter FileNames
    virgin_filenames = [filename for filename in filenames if filename not in [pivot1, pivot2]]

    # load virgin data
    virginAbsFreq = create_csv(virgin_filenames, virgin_filenames, doc_dicts, freq="abs")
    Fwv = create_csv(virgin_filenames, virgin_filenames, doc_dicts, freq="rel")

    # 1:1 merge Fwv with Sw (to discard all disjoint words)
    temp = pd.merge(Fwv, Sw, on='word', how='inner')

    # split filtered Sw
    cleanSw = pd.DataFrame(temp.score)

    # clean up filtered Fwv
    del temp['word']
    del temp['score']
    cleanFwv = temp

    # compute Sv = sum(Fwv * Sw)_{for all w}
    Sv = cleanFwv.T.dot(cleanSw)

    # compute transformed Sv
    Sv_t = (Sv - Sv.mean()) * (Ar.T.std() / Sv.std()) + Sv.mean()

    # compute Vv
    Vv = (cleanFwv * np.square((np.array(cleanSw)
                                - np.array(Sv.T)))).sum(axis=0)

    # 1:1 merge absolute frequencies with Sw (to discard all disjoint words)
    temp = pd.merge(virginAbsFreq, Sw, on='word', how='inner')

    # compute N
    del temp['word']
    del temp['score']
    N = temp.sum(axis=0)

    # compute standard errors and confidence intervals
    std_error = np.sqrt(Vv / N)
    lower = np.array(Sv).flatten() - np.array((1.96 * std_error))
    upper = np.array(Sv).flatten() + np.array((1.96 * std_error))

    # compute transformed confidence intervals
    lower_t = (np.array(lower) - np.array(Sv.mean())) \
              * np.array((Ar.T.std() / Sv.std())) \
              + np.array(Sv.mean())
    upper_t = (np.array(upper) - np.array(Sv.mean())) \
              * np.array((Ar.T.std() / Sv.std())) \
              + np.array(Sv.mean())

    # print everything
    print('Original scores (w/ 95CI):')

    Sv['lower'] = lower
    Sv['upper'] = upper

    print('Transformed scores (w/ 95CI):')
    Sv_t['lower'] = lower_t
    Sv_t['upper'] = upper_t

    score_dict = Sv_t["score"].to_dict()
    score_dict[pivot1] = -1
    score_dict[pivot1] = -1

    # Create Score Dict
    score_dict = Sv["score"].to_dict()
    score_dict = {k + ".txt": v for k, v in score_dict.items()}
    score_dict[pivot1] = -1
    score_dict[pivot2] = 1

    if predictions_file_path:
        io_helper.write_dictionary(predictions_file_path, score_dict)

    return Sv_t["score"].to_dict()


# create function to load and merge data
def create_csv(cases, filenames, doc_dicts, freq="rel"):
    '''
    iterable, string, dict -&gt; pandas.DataFrame
    '''
    output = pd.DataFrame(columns=['word'])

    for case in cases:
        index_case = filenames.index(case)
        if freq == "abs":
            count_freq = doc_dicts[index_case][0]
        elif freq == "rel":
            count_freq = doc_dicts[index_case][1]
        pivot_filename = case.replace('.txt', '')
        new_dict = {"word": [], pivot_filename: []}
        for k, v in count_freq.items():
            new_dict["word"].append(k)
            new_dict[pivot_filename].append(v)

        new_csv = pd.DataFrame(new_dict)

        # merge with previous data
        output = pd.merge(output, new_csv, on='word', how='outer')
        output = output.fillna(0)  # kill NaNs

    return output


def build_vocab(tokens, count_treshold=1):
    print("Building full vocabulary...")
    full_vocab = {}
    for t in tokens:
        if t in full_vocab:
            full_vocab[t] = full_vocab[t] + 1
        else:
            full_vocab[t] = 1

    print("Tresholding vocabulary...")
    vocab = [x for x in full_vocab if full_vocab[x] >= count_treshold]

    total_amount = sum([full_vocab[x] for x in vocab])

    print("Total Number of Words: " + str(total_amount))
    print("Building abostule count dict...")
    counts = {x: full_vocab[x] for x in vocab}

    print("Building relative frequency dict...")
    rel_frequency = {x: full_vocab[x] / total_amount for x in vocab}

    return (counts, rel_frequency)



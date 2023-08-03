import codecs
from scipy import stats
import numpy as np
import sys
import pathlib
import pandas as pd
import math
EU_FOLDERS = ["eu", "non_eu"]
POLICYAREAS = [str(i) for i in range(1, 25)]
GT_NAME = ["scale.txt", "lr_position.txt"]

# here's a simple function to compute pairwise accuracy
def pairwise_accuracy(golds, preds):
    count_good = 0.0
    count_all = 0.0
    for i in range(len(golds) - 1):
        for j in range(i + 1, len(golds)):
            count_all += 1.0
            diff_gold = golds[i] - golds[j]
            diff_pred = preds[i] - preds[j]
            if (diff_gold * diff_pred >= 0):
                count_good += 1.0
    return count_good / count_all


# here's the main function
def evaluate(gold_path, predicted_path):
    # you open the chapel hill referenced positions
    golds = [(x.split()[0].strip(), float(x.split()[1].strip())) for x in
             list(codecs.open(gold_path, "r", "utf-8").readlines())]
    # and normalize them, in case they are not
    max_score = max([x[1] for x in golds])
    min_score = min([x[1] for x in golds])
    diff = float(max_score) - float(min_score)
    golds = [[x[0], (float(x[1]) - float(min_score)) / (diff)] for x in golds]

    # then, you do the same with the scaling positions
    predicts = [(x.split()[0].strip(), float(x.split()[1].strip())) for x in
                list(codecs.open(predicted_path, "r", "utf-8").readlines())]
    max_score = max([x[1] for x in predicts])
    min_score = min([x[1] for x in predicts])
    diff = float(max_score) - float(min_score)
    predicts = [[x[0], (float(x[1]) - float(min_score)) / (diff)] for x in predicts]

    # we check and exclude duplicate files with the same name
    predicts = set(tuple(row) for row in predicts)
    predicts = [list(x) for x in predicts]

    golds = set(tuple(row) for row in golds)
    golds = [list(x) for x in golds]

    file_gold = sorted([x[0] for x in golds])
    file_pred = sorted([x[0] for x in predicts])
    if file_gold != file_pred:
        print("Warning: Gold filenames are not the same as predict names {}.".format(gold_path))
    # we keep only files appearing both in the referenced file and the scaling file
    golds = [x for x in golds if x[0] in [x[0] for x in predicts]]

    predicts = [x for x in predicts if x[0] in [x[0] for x in golds]]

    # we sort them
    golds.sort(key=lambda x: x[0])
    predicts.sort(key=lambda x: x[0])

    gold_scores = [x[1] for x in golds]
    predict_scores = [x[1] for x in predicts]

    # we also considered the correlation with the inverted scaling
    predict_scores_inverse = [1.0 - x[1] for x in predicts]

    pearson = stats.pearsonr(gold_scores, predict_scores)[0]
    spearman = stats.spearmanr(gold_scores, predict_scores)[0]

    pearson_inv = stats.pearsonr(gold_scores, predict_scores_inverse)[0]
    spearman_inv = stats.spearmanr(gold_scores, predict_scores_inverse)[0]

    pa = pairwise_accuracy(gold_scores, predict_scores)
    pa_inv = pairwise_accuracy(gold_scores, predict_scores_inverse)

    # we check whether pearson correlation is higher than inverted pearson, based on this we report the results on the original or inverted scaling
    # you can easily modify this line, to check on one of the other measures or simply reporting the max value for each measure

    if pearson > pearson_inv:
        final_pa = pa
        final_p = pearson
        final_s = spearman
    else:
        final_pa = pa_inv
        final_p = pearson_inv
        final_s = spearman_inv

    return final_pa, final_p, final_s


if __name__ == "__main__":
    gold_path = pathlib.Path(sys.argv[1])
    pred_path = pathlib.Path(sys.argv[2])
    stem_pred_name = pred_path.stem
    all_results = []
    for eu in EU_FOLDERS:
        for policy in POLICYAREAS:
            for gt_name in GT_NAME:
                gold_file = gold_path.joinpath(eu).joinpath(policy).joinpath(gt_name)
                pred_name = stem_pred_name + "_" + eu + "_" + policy + ".txt"
                pred_file = pred_path.joinpath(eu).joinpath(policy).joinpath(pred_name)
                if not pred_file.is_file():
                    print("No Prediction File for {}.".format(pred_file))
                    continue

                if gold_file.is_file():
                    gold_values = [(x.split()[0].strip(), float(x.split()[1].strip())) for x in
                                   list(codecs.open(gold_file, "r", "utf-8").readlines())]

                    if gold_values:
                        # and normalize them, in case they are not
                        max_score = max([x[1] for x in gold_values])
                        min_score = min([x[1] for x in gold_values])
                        diff = float(max_score) - float(min_score)
                        if diff == 0:
                            print(gold_file)
                            continue

                        if sum([math.isnan(i) for i in [x[1] for x in gold_values]]) >= 1:
                            print(gold_file)
                            continue

                        pa, pears, spear = evaluate(gold_file, pred_file)
                        all_results.append([eu, policy, gold_file.stem, pa, pears, spear])
    df = pd.DataFrame(all_results, columns=['eu', 'policyarea', 'dimension', 'pa', 'pears', "spear"])
    print(df)
    print("Calculating mean:")
    mean = df.groupby(["eu", "dimension"])[["pa", "pears", "spear"]].mean()
    print(df.groupby(["eu", "dimension"])[["pa", "pears", "spear"]].mean())
    df.to_csv(pred_path.joinpath("results.csv"))
    mean.to_csv(pred_path.joinpath("summary_results.csv"))

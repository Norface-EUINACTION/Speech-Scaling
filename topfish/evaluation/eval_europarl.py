import codecs
from scipy import stats
import numpy as np
import sys


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
    print(gold_path)
    print(golds)
    # and normalize them, in case they are not
    max_score = max([x[1] for x in golds])
    min_score = min([x[1] for x in golds])
    diff = float(max_score) - float(min_score)
    golds = [[x[0], (float(x[1]) - float(min_score)) / (diff)] for x in golds]
    print(golds)
    # then, you do the same with the scaling positions
    predicts = [(x.split()[0].strip(), float(x.split()[1].strip())) for x in
                list(codecs.open(predicted_path, "r", "utf-8").readlines())]
    print(predicts)
    max_score = max([x[1] for x in predicts])
    min_score = min([x[1] for x in predicts])
    diff = float(max_score) - float(min_score)
    predicts = [[x[0], (float(x[1]) - float(min_score)) / (diff)] for x in predicts]
    print(predicts)
    # we check and exclude duplicate files with the same name
    predicts = set(tuple(row) for row in predicts)
    predicts = [list(x) for x in predicts]
    print(predicts)
    golds = set(tuple(row) for row in golds)
    golds = [list(x) for x in golds]

    # we keep only files appearing both in the referenced file and the scaling file
    golds = [x for x in golds if x[0] in [x[0] for x in predicts]]

    predicts = [x for x in predicts if x[0] in [x[0] for x in golds]]
    print(predicts)
    # we sort them
    golds.sort(key=lambda x: x[0])
    predicts.sort(key=lambda x: x[0])

    gold_scores = [x[1] for x in golds]
    predict_scores = [x[1] for x in predicts]

    # we also considered the correlation with the inverted scaling
    predict_scores_inverse = [1.0 - x[1] for x in predicts]
    print(gold_scores)
    print(predict_scores)
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
    gold_path = sys.argv[1]
    pred_path = sys.argv[2]
    pa, pears, spear = evaluate(gold_path, pred_path)
    print("Pairwise accuracy: " + str(pa))
    print("Pearson coefficient: " + str(pears))
    print("Spearman coefficient: " + str(spear))
import matplotlib.pyplot as plt
import pathlib
import pandas as pd
import argparse
from datetime import datetime
import json
from progress.bar import Bar
import csv
import json
from matplotlib.pyplot import figure
from collections import Counter
OUTPUT_FOLDER = pathlib.Path('visualization')
GT_FILE_NAMES = ["scale.txt"]
EU_FOLDERS = ['eu', 'non_eu']

PARTY_COLOR_MAPPING = {
    "Con": "blue",
    "LibDem": "orange",
    "Lab": "red",
    "SNP": "yellow",
    "Fidesz-KDNP": "orange",
    "MSZP": "red",
    "Jobbik": "green",
    "PS": "pink",
    "PCP": "red",
    "PSD": "orange",
    "CDS-PP": "blue",
}

EU_NONEU_NAMES = {
    "eu": "EU",
    "non_eu": "Domestic"
}

POLICYAREA_MAPPING = {
    1: "Macroeconomics",
    3: "Health",
    5: "Labour",
    6: "Education",
    7: "Environment",
    8: "Energy",
    12: "Law & Crime",
    13: "Social Welfare",
    14: "Housing",
    15: "Domestic Commerce",
    16: "Defence"
}
POLICYAREA = list(POLICYAREA_MAPPING.keys())

PLOT_MAPPING = {
    1: [0, 0],
    3: [0, 1],
    5: [0, 2],
    6: [0, 3],
    7: [1, 0],
    8: [1, 1],
    12: [1, 2],
    13: [1, 3],
    14: [2, 0],
    15: [2, 1],
    16: [2, 2],
}

def main(gt_folder, pred_folder):
    gt_folder = pathlib.Path(gt_folder)
    pred_folder = pathlib.Path(pred_folder)
    model = pred_folder.parent.stem

    country_name = gt_folder.stem
    for eu_folder in EU_FOLDERS:
        plt.rcParams["figure.figsize"] = [20, 16]
        fig, all_axis = plt.subplots(3, 4)
        fig.suptitle(EU_NONEU_NAMES[eu_folder], fontsize=30)

        for policy in POLICYAREA:
            current_output_folder = OUTPUT_FOLDER.joinpath(model).joinpath(country_name).joinpath(eu_folder).joinpath(
                str(policy))
            current_output_folder.mkdir(parents=True, exist_ok=True)
            current_axis = all_axis[PLOT_MAPPING[policy][0], PLOT_MAPPING[policy][1]]
            lines = {}

            # Pred File
            file_name = country_name + "_" + str(eu_folder) + "_" + str(policy) + ".txt"
            pred_file = pred_folder.joinpath(eu_folder).joinpath(str(policy)).joinpath(file_name)
            if not pred_file.is_file():
                continue

            with open(pred_file, newline='') as games:
                pred_csv = csv.reader(games, delimiter='\t')

                for instance in pred_csv:
                    [instance_id, instance_pred] = instance
                    idx = instance_id.split("_")[1].split(".")[0]

                    # Meta GT File
                    meta_name = "meta_" + idx + ".json"
                    meta_file = gt_folder.joinpath(eu_folder).joinpath(str(policy)).joinpath("meta").joinpath(meta_name)
                    with open(meta_file) as json_file:
                        data = json.load(json_file)
                    if data["party"] not in lines:
                        lines[data["party"]] = {"x": [], "y": []}

                    lines[data["party"]]["x"].append(data["period"])
                    lines[data["party"]]["y"].append(float(instance_pred))

            for party_name, values in lines.items():
                sorted_pairs = sorted(zip(values["x"], values["y"]), key=lambda x: x[0])
                if party_name in PARTY_COLOR_MAPPING:
                    current_axis.plot([i[0] for i in sorted_pairs], [i[1] for i in sorted_pairs],
                             label=party_name, marker='o', color=PARTY_COLOR_MAPPING[party_name])

                else:
                    print("No color for party {}".format(party_name))
                    current_axis.plot([i[0] for i in sorted_pairs], [i[1] for i in sorted_pairs],
                                          label=party_name, marker='o')

            # Save Figure
            current_axis.set_xticks(range(1, 4))
            current_axis.set_xlabel("Period")
            current_axis.set_ylabel("Scale")
            current_axis.legend()
            current_axis.title.set_text(POLICYAREA_MAPPING[policy])

            # Save subplots
            extent = current_axis.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
            fig.savefig(current_output_folder.joinpath('single_figure.png'), bbox_inches=extent.expanded(1.2, 1.28))
            current_axis.get_legend().remove()

        all_axis[2, 3].axis('off')
        OUTPUT_FOLDER.joinpath(model).joinpath(country_name).joinpath(eu_folder).mkdir(parents=True, exist_ok=True)

        lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
        lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
        unique_labels = len(set(labels))

        # Remove duplicates
        fig.legend(lines[:unique_labels], labels[:unique_labels], loc=4, prop={'size': 20}, bbox_to_anchor=(0.88, 0.2))

        fig.savefig(OUTPUT_FOLDER.joinpath(model).joinpath(country_name).joinpath(eu_folder).joinpath("scale_period.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('gt_folder', type=str, help='Country name, if more than one tokens, separated with underscore')
    parser.add_argument('pred_folder', type=str,
                        help='Country name, if more than one tokens, separated with underscore')

    args = parser.parse_args()
    main(args.gt_folder, args.pred_folder)

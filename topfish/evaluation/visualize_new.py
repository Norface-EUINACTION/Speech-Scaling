import matplotlib.pyplot as plt
import pathlib
import pandas as pd
import argparse
from datetime import datetime
import json
from progress.bar import Bar
import csv
from datetime import datetime
import numpy as np
import json
import matplotlib.patches as mpatches
import matplotlib.cm as cm
import plotly.graph_objects as go
from matplotlib.pyplot import figure
from collections import Counter
import seaborn as sns
OUTPUT_FOLDER = pathlib.Path('visualization_new')
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

DATE_MAPPING = {
    "united_kingdom": {'1': {'start_date': '06.05.2010', 'end_date': '07.05.2015'},
                       '2': {'start_date': '08.05.2015', 'end_date': '08.06.2017'},
                       '3': {'start_date': '09.06.2017', 'end_date': '31.12.2019'}},

    "uk_annotations": {'1': {'start_date': '06.05.2010', 'end_date': '07.05.2015'},
                       '2': {'start_date': '08.05.2015', 'end_date': '08.06.2017'},
                       '3': {'start_date': '09.06.2017', 'end_date': '31.12.2019'}},

    "austria": {'1': {'start_date': '01.01.2009', 'end_date': '29.09.2013'},
                '2': {'start_date': '30.09.2013', 'end_date': '15.10.2017'},
                '3': {'start_date': '16.10.2017', 'end_date': '31.12.2019'}},

    "portugal": {'1': {'start_date': '27.09.2009', 'end_date': '05.06.2011', 'parties': ['PS', 'PSD', 'PCP']},
                '2': {'start_date': '06.06.2011', 'end_date': '04.10.2015', 'parties': ['PS', 'PSD', 'PCP', 'CDS-PP']},
                '3': {'start_date': '05.10.2015', 'end_date': '31.12.2019', 'parties': ['PS', 'PSD', 'PCP', 'CDS-PP']}},

    "ireland": {'1': {'start_date': '01.01.2009', 'end_date': '25.02.2011',
                      'parties': ['Fine Gael', 'Fianna Fáil', 'Labour Party']},
                '2': {'start_date': '26.02.2011', 'end_date': '26.02.2016',
                      'parties': ['Fine Gael', 'Fianna Fáil', 'Labour Party', 'Sinn Féin']},
                '3': {'start_date': '27.02.2016', 'end_date': '31.12.2019',
                      'parties': ['Fine Gael', 'Fianna Fáil', 'Sinn Féin']}},
    "hungary": {
            '1': {'start_date': '25.04.2010', 'end_date': '06.04.2014', 'parties': ['Fidesz-KDNP', 'MSZP', 'Jobbik']},
            '2': {'start_date': '07.04.2014', 'end_date': '08.04.2018', 'parties': ['Fidesz-KDNP', 'MSZP', 'Jobbik']},
            '3': {'start_date': '09.04.2018', 'end_date': '31.12.2019', 'parties': ['Fidesz-KDNP', 'MSZP', 'Jobbik']}},

    "germany": {
            '1': {'start_date': '27.09.2009', 'end_date': '22.09.2013', 'parties': ['CDU', 'SPD', 'FDP', 'PDS/LINKE']},
            '2': {'start_date': '23.09.2013', 'end_date': '24.09.2017',
                  'parties': ['CDU', 'SPD', 'PDS/LINKE', 'GRUENE']},
            '3': {'start_date': '25.09.2017', 'end_date': '31.12.2019', 'parties': ['CDU', 'SPD', 'FDP', 'AfD']}},

    "france": {
            '1': {'start_date': '01.01.2009', 'end_date': '17.06.2012'},
            '2': {'start_date': '18.06.2012', 'end_date': '18.06.2017'},
            '3': {'start_date': '19.06.2017', 'end_date': '31.12.2019'}},

    "belgium": {
            '1': {'start_date': '01.01.2009', 'end_date': '13.06.2010'},
            '2': {'start_date': '14.06.2010', 'end_date': '25.05.2014'},
            '3': {'start_date': '26.05.2014', 'end_date': '31.12.2019'}}

}


WIDTH_SCALE = 2000

def main(gt_folder, pred_folder):
    gt_folder = pathlib.Path(gt_folder)
    pred_folder = pathlib.Path(pred_folder)
    model = pred_folder.parent.stem

    country_name = gt_folder.stem
    for eu_folder in EU_FOLDERS:
        plt.rcParams["figure.figsize"] = [50, 16]
        plt.rcParams.update({'font.size': 30})

        fig, ax = plt.subplots()
        fig.suptitle(EU_NONEU_NAMES[eu_folder], fontsize=30)

        all_data_per_policyarea = {}
        save_period_scale = []
        all_parties = []
        current_output_folder = OUTPUT_FOLDER.joinpath(model).joinpath(country_name).joinpath(eu_folder)
        current_output_folder.mkdir(parents=True, exist_ok=True)
        for policy in POLICYAREA:
            lines = {}
            # Pred File
            file_name = country_name + "_" + str(eu_folder) + "_" + str(policy) + ".txt"
            pred_file = pred_folder.joinpath(eu_folder).joinpath(str(policy)).joinpath(file_name)
            if not pred_file.is_file():
                continue

            all_data_per_policyarea[policy] = {}

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

            dates = DATE_MAPPING[country_name]

            party_order = sorted(list(lines.keys()))
            date_format = "%d.%m.%Y"
            df_dict = {"Party": party_order}

            total_days = 0
            for key, value in dates.items():
                start_date = value["start_date"]
                end_date = value["end_date"]
                days_diff = datetime.strptime(end_date, date_format) - datetime.strptime(start_date, date_format)
                value["days_total"] = total_days + days_diff.days
                total_days += days_diff.days

            ticks_x = []

            scaling_of_period = []
            for key, value in dates.items():
                percentage = dates[str(key)]["days_total"] / total_days

                df_dict["period_" + str(key)] = [int(percentage * WIDTH_SCALE)] * len(party_order)
                scaling_of_period.append(int(percentage * WIDTH_SCALE))
                ticks_x.append(int(percentage * WIDTH_SCALE))
            save_period_scale = scaling_of_period

            for period_number in reversed(range(1, 4)):

                scales = []
                # palette = sns.dark_palette("#69d", reverse=True, as_cmap=True)
                palette = sns.color_palette("crest", as_cmap=True)
                for party in party_order:
                    if period_number in lines[party]["x"]:
                        index = lines[party]["x"].index(period_number)
                        scale = lines[party]["y"][index]
                        scales.append(palette(scale))
                        all_data_per_policyarea[policy]["scales_" + str(period_number) + "_" + party] = palette(scale)
                    else:
                        scales.append("1.0")

                        all_data_per_policyarea[policy]["scales_" + str(period_number) + "_" + party] = "1.0"
                all_parties.append(df_dict["Party"])

        new_scaling = []
        unflatten_parties = [item for sublist in all_parties for item in sublist]
        unflatten_parties = sorted(list(set(unflatten_parties)))
        for i, scale_list in enumerate(range(len(unflatten_parties))):
            new_scaling = new_scaling + [element+WIDTH_SCALE*i for element in save_period_scale]

        index = 0

        for party_index, party in enumerate(reversed(unflatten_parties)):

            for period in reversed(range(1, 4)):
                create_plot_data = {}
                # Scale Color
                scale_color = []
                # Y Axis
                create_plot_data["Policyarea"] = []
                for policy_index, policy_data in all_data_per_policyarea.items():

                    if "scales_" + str(period) + "_" + party in policy_data:
                        scale_color_value = policy_data["scales_" + str(period) + "_" + party]
                    else:
                        print("Missing data for Policyarea {} and {}".format(POLICYAREA_MAPPING[policy_index], "scales_" + str(period) + "_" + party))
                        scale_color_value = "1.0"
                    scale_color.append(scale_color_value)
                    create_plot_data["Policyarea"].append(POLICYAREA_MAPPING[policy_index])
                # X Axis
                create_plot_data["Period+Party"] = [new_scaling[len(new_scaling)-1-index]]*len(create_plot_data["Policyarea"])
                df = pd.DataFrame(create_plot_data)

                sns.barplot(x="Period+Party", y='Policyarea', data=df, palette=scale_color,
                            orient="h", ax=ax)
                index += 1

            # Plot Seperation Thick Line
            if party_index > 0:
                plt.axvline(x=WIDTH_SCALE*party_index, linewidth=14.0, color="white")

        plt.tick_params(
            axis='x',  # changes apply to the x-axis
            which='major',  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            top=False,
            labelbottom=True)  # labels along the bottom edge are off

        # Labeling of X Axis
        xtick_labels = []
        for period in range(1,4):
            start_year = DATE_MAPPING[country_name][str(period)]["start_date"].split(".")[-1][-2:]
            end_year = DATE_MAPPING[country_name][str(period)]["end_date"].split(".")[-1][-2:]
            xtick_labels.append("’" + str(start_year) + " - " + "’" +str(end_year))

        minor_xticks = []
        x_tick_before = 0
        for tick in new_scaling:
            minor_xticks.append(x_tick_before + (tick - x_tick_before)/2)
            x_tick_before = tick
        xtick_labels = xtick_labels * len(unflatten_parties)

        # Add parties
        party_ticks = []
        for party_index, party in enumerate(unflatten_parties):
            party_ticks.append(WIDTH_SCALE * party_index + (WIDTH_SCALE / 2) + 1)

        minor_xticks = minor_xticks
        xtick_labels = xtick_labels
        ax.set_xticks(party_ticks, minor=False)
        ax.set_xticklabels([r"$\bf{" + party + "}$" for party in unflatten_parties], minor=False)

        ax.set_xticks(minor_xticks, minor=True)
        ax.set_xticklabels(xtick_labels, minor=True)
        ax.set_xlabel("")
        ax.set_ylabel("")

        # setup the colorbar
        nvalues = np.linspace(0,1,100)
        scalarmappaple = cm.ScalarMappable(cmap=palette)
        scalarmappaple.set_array(nvalues)
        position = fig.add_axes([0.9, 0.1, 0.02, 0.8])  ## the parameters are the specified position you set
        plt.colorbar(scalarmappaple, cax=position)

        # Set visibility of ticks & tick labels
        ax.tick_params(axis="x", which="minor", direction="out",
                       top=True, labeltop=True, bottom=False, labelbottom=False)

        print("Save Figure in {}".format(OUTPUT_FOLDER.joinpath(model).joinpath(country_name).joinpath(eu_folder).joinpath("scale_period.png")))
        fig.savefig(OUTPUT_FOLDER.joinpath(model).joinpath(country_name).joinpath(eu_folder).joinpath("scale_period.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('gt_folder', type=str, help='Country name, if more than one tokens, separated with underscore')
    parser.add_argument('pred_folder', type=str,
                        help='Country name, if more than one tokens, separated with underscore')

    args = parser.parse_args()

    main(args.gt_folder, args.pred_folder)

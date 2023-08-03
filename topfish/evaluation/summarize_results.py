import pathlib
import argparse
from datetime import datetime
import pandas as pd

SUMMARY_FILE = "summary_results.csv"


def main(folder: str, countries: list):
    folder = pathlib.Path(folder)

    total_summary = None
    for index, country in enumerate(countries):
        country_summary = folder.joinpath(country).joinpath(SUMMARY_FILE)

        summary_csv = pd.read_csv(country_summary)

        if index == 0:
            total_summary = summary_csv
        else:
            total_summary["pa"] = total_summary["pa"] + summary_csv["pa"]
            total_summary["pears"] = total_summary["pears"] + summary_csv["pears"]
            total_summary["spear"] = total_summary["spear"] + summary_csv["spear"]

    total_summary["pa"] = total_summary["pa"] / len(countries)
    total_summary["pears"] = total_summary["pears"] / len(countries)
    total_summary["spear"] = total_summary["spear"] / len(countries)
    print(total_summary)
    total_summary.to_csv(folder.joinpath("summary_results.csv"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('folder', type=str)

    parser.add_argument('-countries','--list', nargs='+', help='<Required> Set flag', required=True)

    args = parser.parse_args()
    main(args.folder, args.list)

# Author Saad Obaid ul Islam

import pandas as pd
import os
import glob
from io_helper import get_csv_lines
from io_helper import serialize
from datetime import datetime
from io_helper import write_list
from io_helper import write_csv_lines
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--path", default=None, type=str, help="Add base path")
parser.add_argument("--countries", default=None, type=str, help="Add country")



def match_and_add_column(path1, path2, colmatch1, colmatch2, addcol2):
    lines1 = get_csv_lines(path1)
    lines2 = {l[colmatch2] : l[addcol2] for l in get_csv_lines(path2)[1:]}

    nls = [lines1[0] + ["em_party"]]
    for l in lines1[1:]:
        if l[colmatch1] in lines2:
            if lines2[l[colmatch1]].strip() == "NA":
                i = 7
            nls.append(l + ["" if lines2[l[colmatch1]].strip() == "NA" else lines2[l[colmatch1]]])
        else:
            l = l + [""]

    return nls


#check for correct data format
def try_parsing_date(text):
    for fmt in ('%Y-%m-%d', '%d.%m.%Y', '%d/%m/%Y'):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            pass
    raise ValueError('no valid date format found ' + text)



''' Create a dictionary of indices per country. 

To get indices of relevant columns, run the following code


```
df = pd.read_csv({country name}.csv, sep=',')
for idx, col in df.columns:
  print(str(idx) + " : " + col)

'''

def main(args: argparse.Namespace) -> None:
    base_path = args.path
    file_path = args.path

    countries = [args.country]

    indices = {"belgium" : {"speaker" : 7, "date" : 2, "text" : 9, "policyarea" : 11, "partyfacts" : -2, "EP": -1},
               "austria" : {"speaker" : 9, "date" : 4, "text" : 11, "policyarea" : 15, "partyfacts" : -2, "EP": -1},
               "croatia" : {"speaker" : 8, "date" : 3, "text" : 10, "policyarea" : 14, "partyfacts" : -2, "EP": -1},
               "cyprus" : {"speaker" : 5, "dates" : -1, "text" : 7, "policyarea" : 11, "partyfacts" : 16, "EP": -2},
               "finland" : {"speaker" : 7, "date" : 2, "text" : 9, "policyarea" : 13, "partyfacts" : -2, "EP": -1},
               "sweden" : {"speaker" : 8, "date" : 3, "text" : 10, "policyarea" : 15, "partyfacts" : -2, "EP": -1},
               "estonia" : {"speaker" : 7, "date" : 2, "text" : 9, "policyarea" : 14, "partyfacts" : -2, "EP": -1},
               "denmark" : {"speaker" : 7, "date" : 2, "text" : 9, "policyarea" : 13, "partyfacts" : -2, "EP": -1},
               "germany" : {"speaker" : 9, "date" : 4, "text" : 11, "policyarea" : 15, "partyfacts" : -2, "EP": -1},
               "italy" : {"speaker" : 8, "date" : 3, "text" : 10, "policyarea" : 14, "partyfacts" : -2, "EP": -1},
               "ireland" : {"speaker" : 7, "date" : 2, "text" : 9, "policyarea" : 13, "partyfacts" : -2, "EP": -1},
               "hungary" : {"speaker" : 7, "date" : 2, "text" : 9, "policyarea" : 13, "partyfacts" : -2, "EP": -1},
               "portugal" : {"speaker" : 8, "date" : 3, "text" : 10, "policyarea" : 14, "partyfacts" : -2, "EP": -1},
               "spain" : {"speaker" : 8, "date" : 3, "text" : 9, "policyarea" : 13, "partyfacts" : -2, "EP": -1},
               "slovakia" : {"speaker" : 7, "date" : 2, "text" : 9, "policyarea" : 13, "partyfacts" : -2, "EP": -1},
               "france" : {"speaker" : 9, "date" : 4, "text" : 11, "policyarea" : 15, "partyfacts" : -2, "EP": -1},
               "netherlands" : {"speaker" : 8, "date" : 3, "text" : 10, "policyarea" : 14, "partyfacts" : -2, "EP": -1},
               "romania" : {"speaker" : 7, "date" : 2, "text" : 9, "policyarea" : 14, "partyfacts" : -2, "EP": -1},
               "greece" : {"speaker" : 7, "date" : 2, "text" : 9, "policyarea" : 13, "partyfacts" : -2, "EP": -1},
               "czechia" : {"speaker" : 9, "date" : 4, "text" : 11, "policyarea" : 15, "partyfacts" : -2, "EP": -1},
               "malta" : {"speaker" : 7, "date" : 2, "text" : 9, "policyarea" : 13, "partyfacts" : -2, "EP": -1},
               "slovenia" : {"speaker" : 7, "date" : 2, "text" : 9, "policyarea" : 13, "partyfacts" : -2, "EP": -1},
               "latvia" : {"speaker" : 6, "date" : 1, "text" : 8, "policyarea" : 13, "partyfacts" : -2, "EP": -1},
               "poland" : {"speaker" : 6, "date" : 1, "text" : 8, "policyarea" : 12, "partyfacts" : -2, "EP": -1},
               "lithuania": {"speaker" : 7, "date" : 2, "text" : 9, "policyarea" : 13, "partyfacts" : -2, "EP": -1},
               "bulgaria": {"speaker" : 7, "date" : 2, "text" : 9, "policyarea" : 14, "partyfacts" : -2, "EP": -1},
               "norway": {"speaker" : 6, "date" : 1, "text" : 8, "policyarea" : 21, "partyfacts" : -1}
              }

# for each country
    for country in countries:
      #get csv lines and cabinet periods  
      lines = get_csv_lines(os.path.join(file_path, country + ".csv"))[1:]
      aggs = {}
      periods = {l[2] : (datetime.strptime(l[3], '%Y-%m-%d'), datetime.strptime(l[4], '%Y-%m-%d')) for l in get_csv_lines(os.path.join(base_path, "periods.csv"))[1:] if l[0].strip().lower().replace(" ", "_") == country}

      cnt = 0
      print(len(lines))
      found_cmp = False

      # create sub-dictionaries: 
      # per country, per cabinet/period, policy area, per speaker
      for l in lines:
        try:
          if len(l) < 12:
              continue

          cnt += 1
          if cnt % 100000 == 0:
              print(cnt)
          cmp = l[indices[country]["partyfacts"]].replace('.0','')
          date = try_parsing_date(l[indices[country]["date"]].replace("\"", ""))
          per = [p for p in periods if periods[p][0] <= date and periods[p][1] >= date]
          #ep_np = l[indices[country]["EP"]]
          if len(per) > 0:
              period = per[0]
              if period not in aggs:
                  aggs[period] = {}

              polarea = l[indices[country]["policyarea"]]
              if polarea not in aggs[period]:
                  aggs[period][polarea] = {}

              speaker = l[indices[country]["speaker"]].replace("\"", "")
              #ep = l[indices[country]["EP"]]
              party = l[indices[country]["partyfacts"]].replace("\"", "")
              party = (party.replace("/", "-"))

              spp = speaker + "__" + party

              if spp not in aggs[period][polarea]:
                  aggs[period][polarea][spp] = []

              text = l[indices[country]["text"]]
              aggs[period][polarea][spp].append(text)
        except Exception as e:
          print('Exception: ' + str(e))
          print('Count: ' + str(cnt))

      # create sub-folders
      for period in aggs:
        print(period)
        country_dir = os.path.join(base_path,country)
        if not os.path.exists(country_dir):
            os.mkdir(country_dir)
        per_dir = os.path.join(base_path, country, period.replace(" ", "_").lower())
        if not os.path.exists(per_dir):
            os.mkdir(per_dir)

        for pol_area in aggs[period]:
            pa_dir = os.path.join(per_dir, pol_area, "input_files")
            if not os.path.exists(pa_dir):
                os.makedirs(pa_dir)

            for speaker in aggs[period][pol_area]:
                text = " ".join(aggs[period][pol_area][speaker])
                out_path = os.path.join(pa_dir, speaker.replace(" ", "_").replace(".0", "").lower() + ".txt")
                try:
                    write_list(out_path, [text])
                except Exception as e:
                    print("Exception: " + str(e))
                    print("Period: " + str(period))
                    print("Speaker: " + str(speaker))

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)

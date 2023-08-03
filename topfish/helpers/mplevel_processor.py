import argparse
import os
from io_helper import get_csv_lines
from io_helper import serialize
from datetime import datetime
from io_helper import write_list
from io_helper import write_csv_lines

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

	
#lines = match_and_add_column("/ceph/gglavas/data/eia-mp-level-scaling/ep_party_clean_preds.csv", "/ceph/gglavas/data/eia-mp-level-scaling/ep_party_clean_no_preds.csv", 9, 9, -1)
#write_csv_lines("/ceph/gglavas/data/eia-mp-level-scaling/ep_party_clean.csv", lines)
#exit()

base_path = "/ceph/gglavas/data/eia-mp-level-scaling"
indices = {"austria" : {"speaker" : 9, "party" : 10, "date" : 4, "text" : 11, "policyarea" : -5, "cmp_party" : -1},
           "united_kingdom" : {"speaker" : 9, "party" : 10, "date" : 4, "text" : 11, "policyarea" : -6, "cmp_party" : -1},
           "european_parliament" : {"speaker" : 6, "party" : 8, "national_party" : 13, "date" : 1, "text" : 9, "policyarea" : -6, "cmp_party" : 14, "em_party" : -1, "pg_party": 15},
           "portugal" : {"speaker" : -3, "party" : -2, "date" : 1, "text" : 7, "policyarea" : -7, "cmp_party" : -1},
           "germany" : {"speaker" : 10, "party" : 11, "date" : 5, "text" : 12, "policyarea" : -5, "cmp_party" : -1},
           "belgium" : {"speaker" : 9, "party" : 6, "date" : 1, "text" : 7, "policyarea" : 12, "cmp_party" : 10},
           "ireland" : {"speaker" : 9, "party" : 10, "date" : 4, "text" : 11, "policyarea" : -5, "cmp_party" : -1},
           "hungary" : {"speaker" : 9, "party" : 10, "date" : 4, "text" : 11, "policyarea" : -5, "cmp_party" : -1},
           "france" : {"speaker" : -3, "party" : -2, "date" : 2, "text" : 7, "policyarea" : 11, "cmp_party" : -1},
          }

parser = argparse.ArgumentParser()
parser.add_argument('country', type=str,
                        help="Name of the CSV file")

args = parser.parse_args()

periods = {l[2] : (datetime.strptime(l[3], '%d.%m.%Y'), datetime.strptime(l[4], '%d.%m.%Y')) for l in get_csv_lines(os.path.join(base_path, "periods.csv"))[1:] if l[0].strip().lower().replace(" ", "_") == args.country}


if not os.path.isfile(os.path.join(base_path, args.country + "_party_clean.csv")):
    raise argparse.ArgumentError("Input CSV file does not exist")

lines = get_csv_lines(os.path.join(base_path, args.country + "_party_clean.csv"))[1:]
aggs = {}

cnt = 0
print(len(lines))
found_cmp = False

for l in lines:
    if len(l) < 12:
        continue

    cnt += 1
    if cnt % 100000 == 0:
        print(cnt)
    cmp = l[indices[args.country]["cmp_party"]]
    if cmp.strip() == "":
        if (("em_party" not in indices[args.country] or l[indices[args.country]["em_party"]].strip() == "") and
            ("pg_party" not in indices[args.country] or l[indices[args.country]["pg_party"]].strip() == "")):
           
            #print("skipping, no cmp (or any other for EP) party indication")
            continue
    
    if not found_cmp:
        print("Found first CMP in line: " + str(cnt))
        found_cmp = True

    if args.country in ["belgium", "ireland"]:
        l[indices[args.country]["date"]] = l[indices[args.country]["date"]].split("T")[0].strip()

    if args.country in ["hungary"]:
        l[indices[args.country]["date"]] = l[indices[args.country]["date"]].split()[0].strip()

    date = datetime.strptime(l[indices[args.country]["date"]].replace("\"", ""), '%Y-%m-%d' if args.country in ["portugal", "belgium", "ireland", "france"] else '%d/%m/%Y')
    per = [p for p in periods if periods[p][0] <= date and periods[p][1] >= date]
    if len(per) > 0:
        period = per[0]
        if period not in aggs:
            aggs[period] = {}
        
        polarea = l[indices[args.country]["policyarea"]]
        if polarea not in aggs[period]:
            aggs[period][polarea] = {} 

        speaker = l[indices[args.country]["speaker"]].replace("\"", "")
        party = l[indices[args.country]["party"]].replace("\"", "") if args.country != "european_parliament" else (l[indices[args.country]["party"]].replace("\"", "") + "_" + l[indices[args.country]["national_party"]].replace("\"", ""))
        party = party.replace("/", "-")

        spp = speaker + "__" + party

        if spp not in aggs[period][polarea]:
            aggs[period][polarea][spp] = []

        text = l[indices[args.country]["text"]]
        aggs[period][polarea][spp].append(text)

for period in aggs:
    print(period)
    per_dir = os.path.join(base_path, args.country, period.replace(" ", "_").lower())
    if not os.path.exists(per_dir):
        os.mkdir(per_dir)
    
    for pol_area in aggs[period]:
        pa_dir = os.path.join(per_dir, pol_area, "input_files")
        if not os.path.exists(pa_dir):
            os.makedirs(pa_dir)
        
        for speaker in aggs[period][pol_area]:
            text = " ".join(aggs[period][pol_area][speaker])
            out_path = os.path.join(pa_dir, speaker.replace(" ", "_").replace(".", "").lower() + ".txt")
            try:
                write_list(out_path, [text])
            except Exception as e:
                print("Exception: " + str(e))
                print("Period: " + str(period))
                print("Speaker: " + str(speaker))


        

#serialize(aggs, os.path.join(base_path, "austria.pkl"))


#output_path = os.path.join(base_path, "output", args.country + "_" + args.period.replace(" ", "") + str(args.policy))
#if not os.path.exists(output_path): 
#    os.mkdir(output_path)
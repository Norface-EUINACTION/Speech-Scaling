# -*- coding: utf-8 -*-
"""combine_scaling_party.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1HimLgGmQO8Jlb_atdWI66YRhb83H-AlE
"""
import pandas as pd
import os
import glob
import argparse

parser = argparse.ArgumentParser()
#parser.add_argument("--path", default=None, type=str, help="Add base path")
parser.add_argument("--country", default=None, type=str, help="Add country")


def split_file_names(s) -> list:
  x = os.path.splitext(s)[0]
  x = x.split('/')

  return x

def read_file(filename) -> list:
  with open(filename) as file:
    lines = [line.rstrip() for line in file]

  return lines

def main(args: argparse.Namespace) -> None:
  base_path = '/ceph/sobaidul/data/party_scaling/scaling_party' # add base path
  countries=[args.country] # add country
  for c in countries:
    results = list()
    print(c)
    folders = glob.glob(f"{base_path}/{c}/*/*/*-standard.txt")
    for i in folders:
      split_names = split_file_names(i)
      #print(split_names)
      #break
      country = split_names[6]
      cab = split_names[7]
      pol_area = split_names[8]

      contents = read_file(i)
      for idx, j in enumerate(contents):
        partyfacts_score = j.split('\t')
        #mp_cmp = mp_cmp_score[0].split('__')

        #print(mp_cmp)  )
        partyfacts = partyfacts_score[0]
        #CMP = mp_cmp[1]
        #ep = mp_cmp[2]
        score = partyfacts_score[1]

        results.append((cab, pol_area, partyfacts, score))
    #break
    df = pd.DataFrame(results,columns=['cabinet', 'policyarea', 'partyfacts', 'scaled_score'])
    #print(df)
    df.to_csv(f'{base_path}/{c}_party_scaling_scores_normalized.csv', sep=',', index=False)

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
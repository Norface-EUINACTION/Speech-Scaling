# author Saad Obaid ul Islam
import pandas as pd
import os
import glob
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--path", default=None, type=str, help="Add base path")
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


  base_path = args.path # add base path
  countries=[args.country] # add country
  for c in countries:
    results = list()
    print(c)
    folders = glob.glob(f"{base_path}/{c}/*/*/*-standard.txt")
    for i in folders:
      split_names = split_file_names(i)
      #print(split_names)
      #break
      country = split_names[0]
      cab = split_names[1]
      pol_area = split_names[2]

      contents = read_file(i)
      for idx, j in enumerate(contents):
        mp_cmp_score = j.split('\t')
        mp_cmp = mp_cmp_score[0].split('__')
        if len(mp_cmp) <= 1:
          mp = mp_cmp[0]
          score = mp_cmp_score[1]

          results.append((cab,mp,pol_area,'',score))

        else:

          mp = mp_cmp[0]
          CMP = mp_cmp[1]
          #ep = mp_cmp[2]
          score = mp_cmp_score[1]

        results.append((cab,mp,pol_area,CMP,score))
    df = pd.DataFrame(results,columns=['cabinet', 'speaker', 'policyarea', 'partyfacts', 'scaled_score'])
    #print(df)
    df.to_csv(f'{c}_mp_scaling_scores_normalized.csv', sep=',', index=False)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
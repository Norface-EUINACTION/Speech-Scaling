import os
from scaler_sbertscore import scale_sbertscore
import argparse
from helpers import io_helper
import numpy as np
import shutil
from collections import Counter
def normalize_scores(base_path, country):
    country_dir = os.path.join(base_path, args.country)
    period_dirs = [os.path.join(country_dir, x) for x in os.listdir(country_dir) if os.path.isdir(os.path.join(country_dir, x))]
    for per_dir in period_dirs:
        pol_area_dirs = [os.path.join(per_dir, x) for x in os.listdir(per_dir) if os.path.isdir(os.path.join(per_dir, x))]
    
        for input_dir in pol_area_dirs:
            out_path = os.path.join(input_dir, os.path.basename(per_dir) + "-" + os.path.basename(input_dir) + "-scores.txt")

            if os.path.exists(os.path.join(input_dir, "sbert_pairs")):
                print("Removing SBERT pairs")
                shutil.rmtree(os.path.join(input_dir, "sbert_pairs"))
            
            if os.path.exists(os.path.join(input_dir, "embedding")):
                print("Removing Embeddings")
                shutil.rmtree(os.path.join(input_dir, "embedding"))
                
            if not os.path.exists(out_path):
                print("Skipping (no scores produced): " + out_path)
                continue

            out_path_norm = out_path.replace(".txt", "-standard.txt")
            if os.path.exists(out_path_norm):
                print("Skipping (file with normalized scores already exists): " + out_path)
                continue


            print("Normalizing and cleaning: " + out_path)
            lines = io_helper.load_csv_lines(out_path, delimiter='\t')
            
            files = [x[0] for x in lines]
            scores = [float(x[1]) for x in lines]

            st_scores = (scores - np.mean(scores)) / np.std(scores)
            
            lines_to_write = [files[i] + "\t" + str(st_scores[i]) for i in range(len(st_scores))]
            io_helper.write_list(out_path_norm, lines_to_write)

            


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    base_path = ...
    
    parser.add_argument('country', type=str,
                        help='Country name with more that one token should be separated with underscore, e.g. united_kingdom')

    parser.add_argument('language', type=str,
                        help='Language of the country')

    parser.add_argument('clean', type=str,
                        help='Whether to clean SBERT files and normalize scales ("yes") or to run initial scaling ("no")')


    args = parser.parse_args()


    if args.clean == "yes":
        normalize_scores(base_path, args.country)
        exit()

    country_dir = os.path.join(base_path, args.country)

    period_dirs = [os.path.join(country_dir, x) for x in os.listdir(country_dir) if os.path.isdir(os.path.join(country_dir, x))]
    logf = open(f"{base_path}/{args.country}/error.log", "w")
    filef = open(f"{base_path}/{args.country}/unscalable.txt", "w")
    for per_dir in period_dirs:
        pol_area_dirs = [os.path.join(per_dir, x) for x in os.listdir(per_dir) if os.path.isdir(os.path.join(per_dir, x))]
        
        for input_dir in pol_area_dirs:
            out_path = os.path.join(input_dir, os.path.basename(per_dir) + "-" + os.path.basename(input_dir) + "-scores.txt")
            if os.path.exists(out_path):
                print("Skipping (already completed): " + out_path)
                continue

            print("************************************")
            print(os.path.basename(per_dir))
            print(os.path.basename(input_dir))
            print("************************************")

            
            file_names = [os.path.join(input_dir, "input_files", x) for x in os.listdir(os.path.join(input_dir, "input_files"))]
            
            if len(file_names) < 3: 
                print("Skipping: too few files for scaling. File number: " + str(len(file_names))) 
                filef.write("File size: " + str(len(file_names)) + ". Skipping: too few files for scaling for: " + per_dir + '/' + input_dir + "\n")
                continue
            
            print([item for item, count in Counter(file_names).items() if count > 1])
            texts = [io_helper.load_file(x) for x in file_names]

            try:
                scale_sbertscore([os.path.basename(x).split(".")[0] for x in file_names], texts, [args.language] * len(texts), out_path)
                z = 7
            except Exception as e:
                print("Exception: " + str(e))
                logf.write("Cannot open and run the period and policy area {0}: {1}\n\n\n".format(per_dir + '/' + input_dir , str(e)))
            

    filef.close()
    logf.close()

# Politican and Party-level-Scaling
This folder contains files to reproduce the politican and political party level scaling for 28 European parliaments and the European Parliment.

### To do
- [X] Add Preprocessing and Postprocessing scripts
  
- [X] Add Scaling scripts

- [X] Dependencies
      
- [X] Write How-to

# Introduction

This repository contains code for reproducing the politican-level scaling of political speeches from the national parliaments of 28 European countries and the European Parliament. A list of the relevant European countries can be found on the ParlEE Plenary Speeches Harvard Dataverse page [here](https://dataverse.harvard.edu/dataverse/ParlEE). 

# Dependencies
 + python 3.10
 + Pandas
 + NetworkX
 + Numba
 + Sentence Transformers
 + spaCy
 + tqdm


# How-to Politician Level Scaling
1. As the nature of the research focused on comparisions between politican and political parties in European national parliaments and the European Parliament, the first step is to combine text by country from both the European Parliament and the national parliament. A clarifying column is also added to denote the text source either EP (European Parliament) or NP (national parliament). This can be done by running the following code.

```
df = pd.read_csv('<path-to-EP-partyfacts-file>/filename.csv', sep=',')

country = <country name>
lang = <national-language-of-the-country>

temp_df = df.loc[(df['country'] == country) & (df['language_checked'] == lang)]
temp_df['EP'] = ['EP'] * len(temp_df.index)
df_ep = temp_df.reset_index(drop=True)

df_country = pd.read_csv(f'<path-to-country-file>{country.lower()}.csv', sep=',')
df_country['EP'] = ['NP'] * len(df_country.index)
combine_df = pd.concat([df_ep,df_country],join='inner')
combine_df.to_csv(f'<name-of-the-file>.csv', sep=',', index=False)
```
Next step is to edit the data structure in line 64 in the *pre_process.py* file. This data structure is a dictionary of dictionaries that stores indices of relevant columns from the csv file. To get the indices of the relevant columns, run the following code on the `combine_df` dataframe.

```
for idx, i in enumerate(list(combine_df.columns)):
  print(str(idx) + " " + str(i))
```
2. **Preprocess:** We need to split the country csv file in the right format and build a hierarchy of directories. First step is to run the *pre_process_politician.py* by running the command `python3 pre_process_politician.py --path <path-to-country-file> --countries <country-name>`.
This will create a directory structure as follows:
```
country
├── <cabinet_name>
    ├── <policy_area_number>
             ├── input_files
                      ├── <mp_name>_<partyfacts>_<EP_or_NP>.txt
                      ├── <mp_name>_<partyfacts>_<EP_or_NP>.txt
                      ├── <mp_name>_<partyfacts>_<EP_or_NP>.txt
                      ......
    ├── <policy_area_number>
    ├── <policy_area_number>
    ├── <policy_area_number>
    .....
├── <cabinet_name>
    ├── ....
├── <cabinet_name>
     ├── ....
.....
```
3.  **Scaling:** Code for scaling is taken from the original [topfish](https://github.com/codogogo/topfish) repository and modified. First, add the `base_path` in line 53 in the file `topfish/run_scaler_mps.py`. The base path is where the directories created above are stored. Now run `run_scaling.py` by running the command `python2 run_scaling --countries <country-name>`. After the scaling, the directory will have two new files, called `<cabinet>-<policy_area>-scores.txt` and `<cabinet>-<policy_area>-scores-standard.txt` which contain scaled scores of all the MPs in input_files. The directory structure will look as follows now:
```
country
├── <cabinet_name>
    ├── <policy_area_number>
             ├── input_files
                      ├── <mp_name>_<partyfacts>_<EP_or_NP>.txt
                      ├── <mp_name>_<partyfacts>_<EP_or_NP>.txt
                      ├── <mp_name>_<partyfacts>_<EP_or_NP>.txt
                      ......
             ├── `<cabinet>-<policy_area>-scores.txt`
             ├── `<cabinet>-<policy_area>-scores-standard.txt`
    ├── <policy_area_number>
    ├── <policy_area_number>
    ├── <policy_area_number>
    .....
├── <cabinet_name>
    ├── ....
├── <cabinet_name>
     ├── ....
.....
```

4. **Post-process:** The last step is optional if you want scaling results in a nice clean csv file. To do this, run the *post_process_politician.py* `python3 post_process.py --path <path-to-country-file> --countries <country-name>` which will combine the results of all the scaling in one csv file with columns `Cabinet, Speaker, Policyarea, partyfacts, scaled_score`.


# How-to Party Level Scaling

For party-level scaling, we do not need to combine data from the EP with the national parliament data. Here, we simple proceed with the *pre-processing* script. The pre-processing script first drops any row without the `partyfacts` ID and then proceeds with splitting the country csv file.  

1. **Preprocess:** We need to split the country csv file in the right format and build a hierarchy of directories. First step is to run the *pre_process_party.py* by running the command `python3 pre_process_party.py --path <path-to-country-file> --countries <country-name>`.
This will create a directory structure as follows:
```
country
├── <cabinet_name>
    ├── <policy_area_number>
             ├── input_files
                      ├── <partyfacts>.txt
                      ├── <partyfacts>.txt
                      ├── <partyfacts>.txt
                      ......
    ├── <policy_area_number>
    ├── <policy_area_number>
    ├── <policy_area_number>
    .....
├── <cabinet_name>
    ├── ....
├── <cabinet_name>
     ├── ....
.....
```
3.  **Scaling:** Code for scaling is taken from the original [topfish](https://github.com/codogogo/topfish) repository and modified. First, add the `base_path` in line 53 in the file `topfish/run_scaler_mps.py`. The base path is where the directories created above are stored. Now run `run_scaling.py` by running the command `python3 run_scaling --countries <country-name>`. After the scaling, the directory will have two new files, called `<cabinet>-<policy_area>-scores.txt` and `<cabinet>-<policy_area>-scores-standard.txt` which contain scaled scores of all the MPs in input_files. The directory structure will look as follows now:
```
country
├── <cabinet_name>
    ├── <policy_area_number>
             ├── input_files
                      ├── <partyfacts>.txt
                      ├── <partyfacts>.txt
                      ├── <partyfacts>.txt
                      ......
             ├── `<cabinet>-<policy_area>-scores.txt`
             ├── `<cabinet>-<policy_area>-scores-standard.txt`
    ├── <policy_area_number>
    ├── <policy_area_number>
    ├── <policy_area_number>
    .....
├── <cabinet_name>
    ├── ....
├── <cabinet_name>
     ├── ....
.....
```

4. **Post-process:** The last step is optional if you want scaling results in a nice clean csv file. To do this, run the *post_process_party.py* `python3 post_process_party.py --path <path-to-country-file> --countries <country-name>` which will combine the results of all the scaling in one csv file with columns `Cabinet, Policyarea, partyfacts, scaled_score`.


# Citations

  



import nlp
from helpers import io_helper
from scaler_sbertscore import scale_sbertscore, scale_sbertscore_supervised
import os
from datetime import datetime
import nltk
from wordfish import wordfish_scaler
import pathlib
from scaler_wordscore import wordscore

nltk.download('stopwords')

supported_lang_strings = {"en": "english", "fr": "french", "de": "german", "es": "spanish", "it": "italian",
                          "hu": "hungarian", "pt": "portuguese"}


def scale(scaler, data_dir, embeddings, output, language_code, pivot1, pivot2,
          stopwords_path, freqthold, learnrate, trainiters, automatic_pivot):
    if not os.path.isdir(os.path.dirname(data_dir)):
        print("Error: Directory containing the input files not found.")
        exit(code=1)

    if not os.path.isdir(os.path.dirname(output)) and not os.path.dirname(output) == "":
        print("Error: Directory of the output file does not exist.")
        exit(code=1)

    print(datetime.now().strftime('%Y-%m-%d %H:%M:%S') + " Loading files.")
    files = io_helper.load_all_files(data_dir)
    if len(files) < 4:
        print('There need to be at least 4 texts for a meaningful scaling. Skip this.')
        return

    filenames = [x[0] for x in files]
    texts = [x[1] for x in files]

    # Create Automatic Pivot Files for weak supervision
    if automatic_pivot:
        # Assume Evaluation File lies above text folder
        evaluation_file = pathlib.Path(data_dir).parent.joinpath(automatic_pivot)
        if not evaluation_file.exists():
            print('No Evaluation File!')
            return
        with open(evaluation_file) as f:
            eval_list = []
            for line in f:
                line = line.strip("\n")
                eval_list.append([line.split(" ")[0], float(line.split(" ")[1])])
            eval_list = sorted(eval_list, key=lambda x: x[0], reverse=True)
            [pivot1, pivot1_value] = min(eval_list, key=lambda x: x[1])
            [pivot2, pivot2_value] = max(eval_list, key=lambda x: x[1])

    wrong_lang = False
    languages = [x.split("\n", 1)[0].strip().lower() for x in texts]
    texts = [x.split("\n", 1)[1].strip() for x in texts]
    for i in range(len(languages)):
        if languages[i] not in supported_lang_strings.keys() and languages[i] not in supported_lang_strings.values():
            print("The format of the file is incorrect, unspecified or unsupported language: " + str(filenames[i]))
            wrong_lang = True
    if wrong_lang:
        exit(code=2)

    langs = [(l if l in supported_lang_strings.values() else supported_lang_strings[l]) for l in languages]

    stopwords = nltk.corpus.stopwords.words(supported_lang_strings[language_code])

    predictions_serialization_path = output

    if scaler == "scale_efficient":
        nlp.scale_efficient(filenames, texts, langs, embeddings, predictions_serialization_path, stopwords)
    if scaler == "scale":
        nlp.scale(filenames, texts, langs, embeddings, predictions_serialization_path, stopwords)
    elif scaler == "scale_sbertscore":
        scale_sbertscore(filenames, texts, langs, predictions_serialization_path)
    elif scaler == "scale_wordfish":
        wordfish_scaler(data_dir, output, stopwords_path, freqthold, learnrate, trainiters)
    elif scaler == "scale_sbertscore_supervised":
        scale_sbertscore_supervised(filenames, texts, langs, predictions_serialization_path, pivot1, pivot2)
    elif scaler == "scale_wordscore":
        wordscore(filenames, texts, langs, predictions_serialization_path, pivot1, pivot2,
                  pivot1_value, pivot2_value, stopwords)
    elif scaler == "scale_supervised":
        nlp.scale_supervised(filenames, texts, langs, embeddings, predictions_serialization_path, pivot1, pivot2, stopwords)

    print(datetime.now().strftime('%Y-%m-%d %H:%M:%S') + " Scaling completed.", flush=True)

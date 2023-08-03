import pathlib
from embeddings import text_embeddings
from scaler import scale
import argparse
from datetime import datetime
from evaluation.eval_europarl import evaluate

BASE_FOLDER = pathlib.Path(__file__).absolute().parent.parent.parent
OUTPUT_FOLDER = BASE_FOLDER / pathlib.Path('output_scaler/europarl')
INPUT_FOLDER = BASE_FOLDER / pathlib.Path('data/europarl')


def main(scaler: str, legislative_terms: str, language_code: str, embedding_path: str, pivot1: str, pivot2: str):

    idealogy_results = {}
    integration_results = {}

    if language_code.lower() == "all":
        language_code = ["de", "en", "es", "fr", "it"]

    if isinstance(language_code, str):
        language_code = [language_code]

    for language in language_code:
        print("\n\n---------------------Scaling for {}---------------------".format(language.upper()))
        # Build country folder in output folder
        output_folder_scaler = OUTPUT_FOLDER / scaler / legislative_terms / language.upper()
        input_path = INPUT_FOLDER / legislative_terms / language.upper()
        output_folder_scaler.mkdir(parents=True, exist_ok=True)

        if embedding_path:
            # Load embeddings
            print(datetime.now().strftime('%Y-%m-%d %H:%M:%S') + " Loading embeddings.")
            embeddings = text_embeddings.Embeddings()
            embeddings.load_cc_embeddings(embedding_path, limit=200_000, skip_first_line=True)
        else:
            embeddings = None

        output = output_folder_scaler.joinpath("result.txt")

        print(datetime.now().strftime(
            '%Y-%m-%d %H:%M:%S') + " ******** Scaling for {} in {}th legislative_terms term started. ********".format(language, legislative_terms))

        scale(scaler, str(input_path), embeddings, str(output), language, pivot1, pivot2)

        # Evaluation
        pa, pears, spear = evaluate(input_path.parent / "ideology.txt", output)
        idealogy_results[language] = {"pairwise_accuracy": pa, "pearson": pears, "spearman": spear}
        pa, pears, spear = evaluate(input_path.parent / "integration.txt", output)
        integration_results[language] = {"pairwise_accuracy": pa, "pearson": pears, "spearman": spear}

        print_results(idealogy_results, "idealogy")
        print_results(integration_results, "integration")


def print_results(results, stage):
    print("\n\n---------------------Results of {}---------------------".format(stage.upper()))
    for language, result in results.items():
        print("\nResult for Language {}".format(language))
        print("Pairwise accuracy: " + str(result["pairwise_accuracy"]))
        print("Pearson coefficient: " + str(result["pearson"]))
        print("Spearman coefficient: " + str(result["spearman"]))

    print("\nAVERAGE RESULTS:")
    avg_pa = average([result["pairwise_accuracy"] for _, result in results.items()])
    avg_pear = average([result["pearson"] for _, result in results.items()])
    avg_spear = average([result["spearman"] for _, result in results.items()])
    print("Pairwise accuracy: " + str(avg_pa))
    print("Pearson coefficient: " + str(avg_pear))
    print("Spearman coefficient: " + str(avg_spear))


def average(lst):
    return sum(lst) / len(lst)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('scaler', type=str,
                        help='"scale_efficient": Scaling with tf-idf weighting with chosen pretrained embeddings,  \
                             "scale_sbertscore": Scaling with sbert embeddings and bertscore as edge weights \
                             "scale_sbertscore_supervised": Scaling with sbert embeddings with weak supervision')

    parser.add_argument('legislative_terms', type=str,
                        help='Options: "5"th or "6"th legislative terms.')

    parser.add_argument('language_code', type=str,
                        help='Options: "de", "en", "es", "fr", "it", "all"')

    parser.add_argument('--embedding_path', type=str, default=None,
                        help='Needed, if you choose scale_efficient')

    parser.add_argument('--pivot1', type=str, default=None,
                        help='Needed for weak supervision')

    parser.add_argument('--pivot2', type=str, default=None,
                        help='Needed for weak supervision')

    args = parser.parse_args()

    if args.scaler == "scale_efficient" and args.embedding_path is None:
        parser.error("--scale_efficient requires --embedding_path.")
    if args.scaler == "scale_sbertscore_supervised" and (args.pivot1 is None or args.pivot2 is None):
        parser.error("--scale_sbertscore_supervised requires --pivot1 and --pivot2.")

    main(args.scaler, args.legislative_terms, args.language_code, args.embedding_path,
         args.pivot1, args.pivot2)


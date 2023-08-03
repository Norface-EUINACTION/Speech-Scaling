import pathlib
from embeddings import text_embeddings
from scaler import scale
import argparse
from datetime import datetime

OUTPUT_FOLDER = pathlib.Path('output_scaler')
INPUT_FOLDER = pathlib.Path('input_scaler')
EU_FOLDERS = ['eu', 'non_eu']


def main(scaler: str, country: str, language_code: str, categories: list,
         embedding_path: str, pivot1: str, pivot2: str, stopwords, freqthold, learnrate, trainiters, automatic_pivot):
    # Build country folder in output folder
    output_folder_scaler = OUTPUT_FOLDER / scaler
    country_folder = pathlib.Path(__file__).absolute().parent.parent.parent.joinpath(output_folder_scaler).joinpath(
        country)
    country_folder.mkdir(parents=True, exist_ok=True)

    # Build eu and non_eu sub-folders
    for f in EU_FOLDERS:
        f_path = country_folder.joinpath(f)
        f_path.mkdir(parents=True, exist_ok=True)

        # Build policy areas sub-folders
        for i in range(1, 25):
            cap_folder = f_path.joinpath(str(i))
            cap_folder.mkdir(parents=True, exist_ok=True)

    if embedding_path:
        # Load embeddings
        print(datetime.now().strftime('%Y-%m-%d %H:%M:%S') + " Loading embeddings.")
        embeddings = text_embeddings.Embeddings()
        embeddings.load_cc_embeddings(embedding_path, limit=200_000, skip_first_line=True)
    else:
        embeddings = None

    for f in EU_FOLDERS:
        print(datetime.now().strftime('%Y-%m-%d %H:%M:%S') + f" Starting with {f} related speeches.")
        output_path = country_folder.joinpath(f)
        input_path = pathlib.Path(__file__).absolute().parent.parent.parent.joinpath(INPUT_FOLDER).joinpath(
            country).joinpath(f)
        if categories:
            iterate_over_categories = [input_path.joinpath(x) for x in categories]
        else:
            # Iterate over all
            iterate_over_categories = input_path.iterdir()

        for cap in iterate_over_categories:
            docs = cap.joinpath('text')

            filename_output = country + '_' + f + '_' + cap.stem + '.txt'
            output = output_path.joinpath(cap.stem).joinpath(filename_output)

            print(datetime.now().strftime(
                '%Y-%m-%d %H:%M:%S') + " ******** Scaling for cap = " + cap.stem + " started. ********")

            scale(scaler, str(docs), embeddings, str(output), language_code, pivot1, pivot2, 
                  stopwords, freqthold, learnrate, trainiters, automatic_pivot)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('scaler', type=str,
                        help='"scale_efficient": Scaling with tf-idf weighting with chosen pretrained embeddings,  \
                                 "scale_sbertscore": Scaling with sbert embeddings and bertscore as edge weights \
                                 "scale_sbertscore_supervised": Scaling with sbert embeddings with weak supervision')

    parser.add_argument('country', type=str,
                        help='Country name with more that one token should be separated with underscore, e.g. united_kingdom')

    parser.add_argument('language_code', type=str,
                        help='Options: "de", "en", "es", "fr", "it", "all"')

    parser.add_argument('--categories', type=str, default=None,
                        help='Categories to iterate through. None iterates over all. Format as "2 3 4" .')

    parser.add_argument('--embedding_path', type=str, default=None,
                        help='Needed, if you choose scale_efficient')

    parser.add_argument('--pivot1', type=str, default=None,
                        help='Needed for weak supervision')

    parser.add_argument('--pivot2', type=str, default=None,
                        help='Needed for weak supervision')

    parser.add_argument('--automatic_pivot', type=str, default=None,
                        help='Optional for weak supervision: Given annotation txt file NAME, calculate automatically pivot1 and pivot2.')

    parser.add_argument('--stopwords', type=str, default=None,
                        help='Needed for wordfish')

    parser.add_argument('--freqthold', type=str, default=None,
                        help='Needed for wordfish')

    parser.add_argument('--learnrate', type=str, default=None,
                        help='Needed for wordfish')

    parser.add_argument('--trainiters', type=str, default=None,
                        help='Needed for wordfish')

    args = parser.parse_args()

    if args.scaler == "scale_efficient" and args.embedding_path is None:
        parser.error("--scale_efficient requires --embedding_path.")
    if args.scaler == "scale" and args.embedding_path is None:
        parser.error("--scale requires --embedding_path.")
    if args.scaler == "scale_sbertscore_supervised" and (args.pivot1 is None or args.pivot2 is None) and args.automatic_pivot is None:
        parser.error("--scale_sbertscore_supervised requires --pivot1 and --pivot2.")
    if args.categories:
        categories = args.categories.split(" ")
    else:
        categories = None
    main(args.scaler, args.country, args.language_code, categories, args.embedding_path, args.pivot1,
         args.pivot2, args.stopwords, args.freqthold, args.learnrate, args.trainiters, args.automatic_pivot)





def scale_sbertscore(filenames, texts, languages, embeddings, predictions_file_path, parameters, emb_lang='default',
                    stopwords=[]):
    """Scaling with tf-idf weighting with embeddings (created on the fly).

    Args:
        filenames ():
        texts ():
        languages ():
        embeddings ():
        predictions_file_path ():
        parameters ():
        emb_lang ():
        stopwords ():

    Returns:

    """
    print(datetime.now().strftime('%Y-%m-%d %H:%M:%S') + " Tokenizing documents.", flush=True)
    texts_tokenized = []
    for i in range(len(texts)):
        # print("Document " + str(i + 1) + " of " + str(len(texts)), flush = True)
        texts_tokenized.append(simple_sts.simple_tokenize(texts[i], stopwords, lang_prefix=None))

    print(datetime.now().strftime('%Y-%m-%d %H:%M:%S') + " Building tf-idf indices for weighted aggregation.",
          flush=True)
    tf_index, idf_index = simple_sts.build_tf_idf_indices(texts_tokenized)

    agg_vecs = []
    zero_vectors_idx = []
    for i in range(len(texts_tokenized)):
        # print("Aggregating vector of the document: " + str(i+1) + " of " + str(len(texts_tokenized)), flush = True)
        agg_vec = simple_sts.aggregate_weighted_text_embedding(embeddings, tf_index[i], idf_index, emb_lang,
                                                               weigh_idf=(len(set(languages)) == 1))
        agg_vecs.append(agg_vec)

        # Dealing with all-zero embeddings
        if np.all(np.array(agg_vec) == 0):
            print(f'Warning: Vector with zeros: {filenames[i]}')
            zero_vectors_idx.append(i)

    ## Remove from filenames and agg_vecs
    for j in sorted(zero_vectors_idx, reverse=True):
        del filenames[j]
        del agg_vecs[j]

    agg_vecs = np.vstack(agg_vecs)
    agg_vecs = common_component_removal(agg_vecs, pc=1)

    print(datetime.now().strftime('%Y-%m-%d %H:%M:%S') + " Computing pairwise similarities.", flush=True)
    pairs = simple_sts.fast_cosine_similarity(agg_vecs, filenames)
    with open("/work-ceph/mkuc/eia-crawling/eia_crawling/sim_pairs_accelerated.csv", "w") as f:
        writer = csv.writer(f)
        for row in pairs:
            writer.writerow(row)

    # rescale distances and produce similarities
    print(datetime.now().strftime('%Y-%m-%d %H:%M:%S') + " Normalizing pairwise similarities.", flush=True)
    max_sim = max([x[2] for x in pairs])
    min_sim = min([x[2] for x in pairs])
    pairs = [(x[0], x[1], (x[2] - min_sim) / (max_sim - min_sim)) for x in pairs]

    print(datetime.now().strftime('%Y-%m-%d %H:%M:%S') + " Fixing the pivot documents for scaling.", flush=True)
    min_sim_pair = [x for x in pairs if x[2] == 0][0]
    fixed = [(filenames.index(min_sim_pair[0]), -1.0), (filenames.index(min_sim_pair[1]), 1.0)]

    # propagating position scores, i.e., scaling
    print(datetime.now().strftime(
        '%Y-%m-%d %H:%M:%S') + " Running graph-based label propagation with pivot rescaling and score normalization.",
          flush=True)
    g = graph.Graph(nodes=filenames, edges=pairs)
    scores = g.harmonic_function_label_propagation(fixed, rescale_extremes=True, normalize=True)

    if predictions_file_path:
        io_helper.write_dictionary(predictions_file_path, scores)

    return scores
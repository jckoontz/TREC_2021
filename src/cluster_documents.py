'''
Cluster sentence embeddings for TREC Clinical Trials documents
'''

import os
import sys
import json
import yaml
import argparse

from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
from tqdm import tqdm


def main(args=None):
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    config = load_config(args.config_path)

    embeddings, topic_ids = load_sentence_embeddings(
        config['clustering']['embedding_paths'])

    kmeans_model = train_kmeans_model(
        config['clustering']['model_params'], embeddings)

    clusters_df = get_cluster_df(kmeans_model, topic_ids)

    save_clusters_df(clusters_df, config['clustering']
                     ['output_path'], config['clustering']['filename'])


def load_config(config_path: str) -> dict:
    '''
    Load configuration file
    '''
    with open(config_path) as f:
        try:
            config = yaml.safe_load(f)
        except:
            raise FileNotFoundError('Could not find configuration')
    return config


def load_sentence_embeddings(embedding_paths: list) -> np.ndarray:
    embeddings, topic_ids = [], []
    for path in tqdm(embedding_paths):
        loaded_npz = np.load(path, allow_pickle=True)
        embeddings.extend(loaded_npz['embeddings'])
        topic_ids.extend(loaded_npz['topics'])
    return np.array(embeddings), topic_ids


def train_kmeans_model(model_params: dict, sentence_embeddings):
    model = KMeans(**model_params).fit(sentence_embeddings)
    return model


def get_cluster_df(model, topic_ids):
    df_clusters = pd.DataFrame(
        {'topic_id': topic_ids, 'cluster': model.labels_})
    return df_clusters


def save_clusters_df(df: pd.DataFrame, outpath: str, fname: str):
    df.to_csv(os.path.join(outpath, fname))
    print(f'DataFrame saved as {fname} to {outpath}')


def parse_args(args):
    epilog = ''
    description = ''
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=epilog)
    parser.add_argument('--config_path', help='Path to the config', type=str)
    return parser.parse_args(args)


if __name__ == '__main__':
    main()

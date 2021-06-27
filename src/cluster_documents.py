'''
Cluster sentence embeddings for TREC Clinical Trials documents
'''

import os
import sys
from typing import Any
import yaml
import argparse

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
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

    if config['clustering']['model_type'] == 'kmeans':
        model = train_kmeans_model(
            config['clustering']['kmeans']['model_params'], embeddings)
    else:
        model = train_gmm_model(
            config['clustering']['gmm']['model_params'], embeddings)

    clusters_df = get_cluster_df(
        model, topic_ids, config['clustering']['model_type'], embeddings)

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
    '''
    Load precomputed sentence embeddings
    '''
    embeddings, topic_ids = [], []
    for path in tqdm(embedding_paths):
        loaded_npz = np.load(path, allow_pickle=True)
        embeddings.extend(loaded_npz['embeddings'])
        topic_ids.extend(loaded_npz['topics'])
    return np.array(embeddings), topic_ids


def train_kmeans_model(model_params: dict, sentence_embeddings: np.ndarray):
    '''
    Fit kmeans model
    '''
    model = KMeans(**model_params, verbose=1).fit(sentence_embeddings)
    return model


def train_gmm_model(model_params: dict, sentence_embeddings: np.ndarray):
    '''
    Fit GMM 
    '''
    model = GaussianMixture(**model_params).fit(sentence_embeddings)
    return model


def get_cluster_df(model: Any, topic_ids: list, model_type: str, sentence_embeddings: np.ndarray):
    '''
    Create DataFrame of topic_ids and corresponding cluster labels
    '''
    if model_type == 'kmeans':
        df_clusters = pd.DataFrame(
            {'topic_id': topic_ids, 'cluster': model.labels_})
    else:
        df_clusters = pd.DataFrame(
            {'topic_id': topic_ids, 'cluster': model.predict(sentence_embeddings)})
    return df_clusters


def save_clusters_df(df: pd.DataFrame, outpath: str, fname: str):
    '''
    Save cluster DataFrame to csv
    '''
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

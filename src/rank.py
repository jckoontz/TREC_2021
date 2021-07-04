'''
Rank topics for TREC 2021 Clinical Trials Track
'''

import os
import sys
import argparse
import json
import yaml
from collections import OrderedDict
import numpy as np
from tqdm import tqdm
from sentence_transformers import util
import torch


def main(args=None):

    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    config = load_config(args.config_path)

    document_embeddings, document_ids = get_document_embeddings(
        config['rank']['document_embeddings'])
    topic_embeddings, topic_ids = get_topic_embeddings(
        config['rank']['topic_embeddings'])

    rank_documents(document_embeddings, document_ids,
                   topic_embeddings, topic_ids, config['rank'])


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


def get_document_embeddings(path: str) -> tuple:
    '''
    Loads pretrained document embeddings of Clinical Descriptions
    '''
    embeddings = np.load(path, allow_pickle=True)
    return torch.from_numpy(embeddings['embeddings']), embeddings['document_ids']


def get_topic_embeddings(path: str) -> tuple:
    '''
    Get topic embeddings and sort by id for final results output
    '''
    topics = {}
    embeddings = np.load(path, allow_pickle=True)
    topic_embeddings = embeddings['embeddings']
    topic_ids = embeddings['document_ids']

    for id, embedding in tqdm(zip(topic_ids, topic_embeddings)):
        topics[int(id)] = embedding

    sorted_embeddings = []
    sorted_topic_ids = []

    for key in sorted(topics.keys()):
        sorted_embeddings.append(topics[key])
        sorted_topic_ids.append(str(key))

    return torch.from_numpy(np.array(sorted_embeddings)), np.array(sorted_topic_ids)


def rank_documents(document_embeddings: torch.Tensor, document_ids: np.ndarray, topic_embeddings: torch.Tensor, topic_ids: np.ndarray, config: dict):
    '''
    Format for results is defined as: 
    TOPIC_NO Q0 ID RANK SCORE RUN_NAME
    '''
    run_name = config['run_name']
    for topic, topic_id in tqdm(zip(topic_embeddings, topic_ids)):
        cos_scores = util.pytorch_cos_sim(topic, document_embeddings)[0]
        top_results = torch.topk(cos_scores, k=config['k'])
        i = 1
        for score, idx in zip(top_results[0], top_results[1]):
            doc_idx = document_ids[idx]
            entry = f'{topic_id} Q0 {doc_idx} {i} {score:2f} {run_name}\n'
            with open(config['output_path'], 'a+') as file:
                file.write(entry)
                file.close()
            i += 1


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

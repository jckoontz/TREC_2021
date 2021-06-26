'''
Create sentence vectors with transformers for Trec2021 Clinical Track
'''
import os
import sys
import json
import yaml
import argparse
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm


def main(args=None):
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    config = load_config(args.config_path)

    labeled_documents = get_labeled_documents(config)

    sentences = get_sentences(labeled_documents)

    document_ids = sorted(labeled_documents.keys())

    model, tokenizer = get_model_and_tokenizer(config['sentence_encoder'])

    embeddings = get_embeddings(model, tokenizer, sentences)

    save_embeddings(config['embeddings'],
                    embeddings, document_ids)

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


def get_labeled_documents(config: dict) -> dict:
    '''
    Load labeled clinical trials json files and return single dict
    '''
    labeled_documents = {}
    for _, path in config['clinical_trials']['labeled'].items():
        with open(path, 'r') as f:
            labeled_document = json.load(f)
            f.close()
        labeled_documents.update(labeled_document)
    return labeled_documents


def get_sentences(labeled_documents: dict) -> list:
    '''
    Get list of sentences to embed
    '''
    sentences = []
    for key in tqdm(sorted(labeled_documents.keys())):
        tokens = labeled_documents[key]['tokens']
        sent = ' '.join(tok for tok in tokens)
        sentences.append(sent)
    return sentences


def get_model_and_tokenizer(model_conf: dict) -> tuple:
    '''
    Load model and tokenizer
    '''
    model = AutoModel.from_pretrained(model_conf['model'])
    tokenizer = AutoTokenizer.from_pretrained(model_conf['model'])
    return model, tokenizer


def get_embeddings(model: AutoModel, tokenizer: AutoTokenizer, sentences: list) -> np.ndarray:
    '''
    Calculate sentence embeddings
    '''
    embeddings = []
    for sent in tqdm(sentences):
        encoded_input = tokenizer(
            sent, padding=True, truncation=True, max_length=64, return_tensors='pt')
        with torch.no_grad():
            model_output = model(**encoded_input)
            embedding = model_output.pooler_output
            embedding = torch.nn.functional.normalize(embedding)
            embeddings.append(embedding.detach().cpu().numpy()[0])
    return np.vstack([embeddings])


def save_embeddings(embeddings_config: dict, embeddings: np.ndarray, document_ids: list):
    '''
    Save sentence embeddings as .npz
    '''
    np.savez_compressed(os.path.join(embeddings_config['output_path'], embeddings_config['filename']), embeddings=embeddings,
                                     document_ids=document_ids)


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

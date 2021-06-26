'''
Apply i2b2 2010 NER model to TREC 2021 documents
'''

import os
import sys
import argparse
import yamlsss
import json
import torch
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import BertForTokenClassification, BertTokenizer
from scipy.special import softmax
import numpy as np
import tqdm

def main(args=None):
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    config = load_config(args.config_path)

    model, tokenizer = load_ner_model(args.model_path, len(config['ner']['tag2idx']))

    clinical_descriptions = load_clinical_descriptions(args.clinical_descriptions)
    print(len(clinical_descriptions.keys()))

    tag_sentences(clinical_descriptions, model, tokenizer, args.output_path, config)


def load_config(config_path:str) -> dict:
    '''
    Load configuration file
    '''
    with open(config_path) as f:
        try:
            config = yaml.safe_load(f)
        except:
            raise FileNotFoundError('Could not find configuration')
    return config   


def load_ner_model(model_path: str, num_labels: int):
    '''
    Loads trained i2b2 2010 ner model + tokenizer
    '''
    tokenizer = BertTokenizer.from_pretrained(model_path,do_lower_case=False)
    model = BertForTokenClassification.from_pretrained(model_path,num_labels=num_labels)
    return model, tokenizer


def load_clinical_descriptions(file_path: str):
    '''
    Loads clinical descriptions json which contains
    sentences to label
    '''
    with open(file_path, 'r') as file:
        try:
            clinical_descriptions = json.load(file)
        except:
            raise FileNotFoundError('Clinical trials json not found')
        return clinical_descriptions


def tag_sentences(clinical_descriptions:dict, model, tokenizer, path:str, config:dict):
    '''
    Classify NER concepts in clinical descriptions textss
    '''
    tag2name={config['ner']['tag2idx'][key] : key for key in config['ner']['tag2idx'].keys()}
    for description_id in tqdm.tqdm(clinical_descriptions.keys()):
        sent, label = make_inference(clinical_descriptions[description_id], model, tokenizer, 
        config['ner']['max_len'], tag2name)
        t_map, l_map = detokenize(sent, label)
        keys = list(t_map.keys())
        keys.sort()
        tokens, labels = [], []
        for k in keys:
            if l_map[k].startswith('B') or l_map[k].startswith('I'):
                tokens.append(t_map[k])
                labels.append(l_map[k])

        with open(path, 'r+') as file:
            labeled_clinical_trials = json.load(file)
            entry = {description_id: {'tokens': tokens, 
                     'labels': labels}}
            labeled_clinical_trials.update(entry)
            file.seek(0)
            json.dump(labeled_clinical_trials, file)
            file.close()


def make_inference(text:str, model, tokenizer, max_len:int, tag2name:dict): 
  '''
  Make NER predictions
  '''
  test_query = text

  tokenized_texts = []
  temp_token = []
  temp_token.append('[CLS]')
  token_list = tokenizer.tokenize(test_query)

  for m,token in enumerate(token_list):
      temp_token.append(token)
  
  # Trim the token to fit the length requirement
  if len(temp_token) > max_len-1:
      temp_token= temp_token[:max_len-1]
  temp_token.append('[SEP]')
  tokenized_texts.append(temp_token)
  input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts],
                            maxlen=max_len, dtype="long", truncating="post", padding="post")

  # For fine tune of predict, with token mask is 1,pad token is 0
  attention_masks = [[int(i>0) for i in ii] for ii in input_ids]
  segment_ids = [[0] * len(input_id) for input_id in input_ids]
  input_ids = torch.tensor(input_ids)
  attention_masks = torch.tensor(attention_masks)
  segment_ids = torch.tensor(segment_ids)
  model.eval()

  with torch.no_grad():
          outputs = model(input_ids, token_type_ids=None,
          attention_mask=None,)
          logits = outputs[0]
  predict_results = logits.detach().cpu().numpy()

  result_arrays_soft = softmax(predict_results[0])

  result_array = result_arrays_soft

  result_list = np.argmax(result_array,axis=-1)
  sents, labels = [], []
  
  for i, mark in enumerate(attention_masks[0]):
      if mark>0:
          sents.append(temp_token[i])
          labels.append(tag2name[result_list[i]])

  return sents, labels


def detokenize(sent, lab):
  '''
  @TODO improve this function
  Corrects BERT tokenizations
  '''
  sent.pop(0)
  lab.pop(0)
  sent.pop(-1)
  lab.pop(-1)
  assert len(sent) == len(lab)
  tok_map = dict()
  labels_map = dict()
  out_toks = []
  for index, _ in enumerate(sent):

      if (index+1 < len(sent)):
        if not sent[index].startswith('#') and not sent[index+1].startswith('#'):
          tok_map[index] = sent[index]
          labels_map[index] = lab[index]
        else:
          if not sent[index].startswith('#'):
            labels_map[index] = lab[index]
            out_toks.append(index)
      else:
        if not sent[index].startswith('#'):
            labels_map[index] = lab[index]
            out_toks.append(index)

  for tok in out_toks:
    s = sent[tok]
    for i in range(tok+1, len(sent)):
      if not sent[i].startswith('#'):
        break
      else:
        s += sent[i].strip('##')
    tok_map[tok] = s 

  return tok_map, labels_map


def parse_args(args):
    epilog = """
    Examples of usage

    python detokenize.py sentences_to_predict output_path model_path
    """
    description = """
    Makes NER inference on a list of sentences. 
    Detokenizes wordpiece tokens
    """
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=epilog)
    parser.add_argument('--config_path', help='Path to the config', type=str)
    parser.add_argument('--clinical_descriptions', help='Path to the clinical descriptions to label', type=str)
    parser.add_argument('--model_path',
                        help='Path to the model for making inferences',
                        type=str)
    parser.add_argument('--output_path', help='path to write tagged sentences',
                        type=str)
    return parser.parse_args(args)

if __name__ == '__main__':
    main()

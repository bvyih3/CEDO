import json
import numpy as np
import os
import sys
sys.path.append(os.getcwd())

import utils.config as config
from utils.dataset import Dictionary
import argparse
from main_arcface import parse_args
    
def create_dictionary(dataroot):
    dictionary = Dictionary()
    questions = []
    files = [
        'train.json',
        # 'val.json',
        'test.json',
    ]
    # files = [
    #     'train',
    #     'hard',
    #     'easy',
    #     'counterexample',
    #     'all'
    # ]
    for path in files:
        # path += '_targets.json'
        question_path = os.path.join(dataroot, path)
        print(question_path)
        qs = json.load(open(question_path))
        for q in qs:
            dictionary.tokenize(q['question'], True, True)
            dictionary.tokenize(q['answer'], True, False)
            # for a, s in q['label'].items():
            #     dictionary.tokenize(str(a), True, False)
    print('len:', len(dictionary))
    return dictionary


def create_glove_embedding_init(idx2word, glove_file):
    """ Using pre-trained glove embedding for questions. """
    word2emb = {}
    with open(glove_file, 'r') as f:
        entries = f.readlines()
    emb_dim = len(entries[0].split(' ')) - 1
    print('pre-trained embedding dim is {}d'.format(emb_dim))
    weights = np.zeros((len(idx2word), emb_dim), dtype=np.float32)

    for entry in entries:
        vals = entry.split(' ')
        word = vals[0]
        vals = map(float, vals[1:])
        word2emb[word] = np.array(list(vals))
    for idx, word in enumerate(idx2word):
        if word not in word2emb:
            continue
        weights[idx] = word2emb[word]
    return weights, word2emb

if __name__ == '__main__':
    args = parse_args()
    print(args)
    dataset = args.dataset
    config.dataset = dataset
    config.update_paths(args.dataset)

    print(config.qa_path)
    d = create_dictionary(config.qa_path)
    d.dump_to_file(config.dict_path)
    print(config.dict_path)
    d = Dictionary.load_from_file(config.dict_path)
    print(config.glove_path)
    # print('len:', len(d.word2idx))
    # print('len1:', len(d.idx2word))
    # for word in d.idx2word:
    #     if str(word) not in d.word2idx.keys():
    #         print(word, '!!!')
    # for word in d.word2idx.keys():
    #     if str(word) not in d.idx2word:
    #         print(word, '???')
    weights, word2emb = create_glove_embedding_init(d.idx2word, config.glove_path)
    print(config.glove_embed_path)
    np.save(config.glove_embed_path, weights)

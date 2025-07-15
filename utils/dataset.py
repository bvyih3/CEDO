import os
import json
import torch
import random
import h5py
import numpy as np
import utils.utils as utils 
from torch.utils.data import Dataset
import utils.config as config
import copy
import torch.nn.functional as F

torch.utils.data.ConcatDataset.__getattr__ = lambda self, attr: getattr(self.datasets[0], attr)


class Dictionary(object):
    def __init__(self, word2idx=None, idx2word=None):
        if word2idx is None:
            word2idx = {}
        if idx2word is None:
            idx2word = []
        self.word2idx = word2idx
        self.idx2word = idx2word

    @property
    def ntoken(self):
        return len(self.word2idx)

    @property
    def padding_idx(self):
        return len(self.word2idx)

    def tokenize(self, sentence, add_word, isQue):
        words = sentence
        if isQue:
            sentence = sentence.lower()
            sentence = sentence.replace(
                ',', '').replace('?', '').replace('\'s', ' \'s')
            words = sentence.split()
        tokens = []
        if add_word:
            if isQue:
                for w in words:
                    tokens.append(self.add_word(w))
            else:
                tokens.append(self.add_word(words))
        else:
            for w in words:
                tokens.append(self.word2idx[w])
        return tokens

    def dump_to_file(self, path):
        json.dump([self.word2idx, self.idx2word], open(path, 'w'))
        print('dictionary dumped to {}'.format(path))

    @classmethod
    def load_from_file(cls, path):
        print('loading dictionary from {}'.format(path))
        word2idx, idx2word = json.load(open(config.dict_path, 'r'))
        d = cls(word2idx, idx2word)
        return d

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)

def _create_entry(img, question, answer):
    answer.pop('image_id')
    answer.pop('question_id')
    entry = {
        'question_id': question['question_id'],
        'image_id': question['image_id'],
        'image': img,
        'question': question['question'],
        'answer': answer,
        'answer_type': question['answer_type']}
    return entry


def _load_dataset(cache_path, name, img_id2val,ratio=1.0):
    """ Load entries. img_id2val: dict {img_id -> val} ,
        val can be used to retrieve image or features.
    """
    if 'vqace' in cache_path:
        prefix = name
        if name == 'cou':
            prefix = 'counterexample'
        prefix += '_targets'
        question_path = os.path.join(config.main_path, prefix + '.json')
    else:
        question_path = os.path.join(config.main_path, name + '.json')
    print(question_path)
    questions = json.load(open(question_path, 'r'))
    questions = sorted(questions, key=lambda x: x['question_id'])
    answer_path = os.path.join(cache_path, '{}_target.json'.format(name))
    print(answer_path)
    answers = json.load(open(answer_path, 'r'))
    answers = sorted(answers, key=lambda x: x['question_id'])
    utils.assert_eq(len(questions), len(answers))
    if ratio < 1.0:
        print('--------ratio-----------',ratio)
        # sampling traing instance to construct smaller training set.
        index = random.sample(range(0, len(questions)), int(len(questions) * ratio))
        questions = [questions[i] for i in index]
        answers = [answers[i] for i in index]

    entries = []
    for question, answer in zip(questions, answers):
        if 'img_id' in question:
            question['image_id'] = question['img_id']
            question.pop('img_id')
        elif 'image id' in question:
            question['image_id'] = question['image id']
            question.pop('image id')
        if 'sent' in question:
            question['question'] = question['sent']
            question.pop('sent')
        if name == 'train':
            utils.assert_eq(question['question_id'], answer['question_id'])
            utils.assert_eq(question['image_id'], answer['image_id'])
            if 'vqace' in question_path:
                question['image_id'] = int(question['image_id'].split('_')[2].lstrip('0'))
        if 'answer_type' not in question:
            question['answer_type'] = None
        question['image_id'] = str(question['image_id'])
        img_id = question['image_id']
        entries.append(_create_entry(img_id2val[img_id], question, answer))
    return entries


def _load_margin(cache_path, name):
    """ Load answer margin per question type.
    """

    mask_path = os.path.join(cache_path, '{}_margin.json'.format(name))
    print(mask_path)
    qt_dict = json.load(open(mask_path, 'r'))
    for qt in qt_dict:
        ans_num_dict = utils.json_keys2int(qt_dict[qt])
        ans = torch.tensor(list(ans_num_dict.keys()), dtype=torch.int64)
        portion = torch.tensor(list(ans_num_dict.values()), dtype=torch.float32)
        qt_dict[qt] = (ans, portion)

    mask_path = os.path.join(cache_path, '{}_freq.json'.format(name))
    qt_dict_freq = json.load(open(mask_path, 'r'))

    qt_cnt = {}
    qt_ans_cnt = {}
    qt_scale = copy.deepcopy(qt_dict_freq)

    for qt in qt_dict_freq:
        ans_num_dict = utils.json_keys2int(qt_dict_freq[qt])
        ans = torch.tensor(list(ans_num_dict.keys()), dtype=torch.int64)
        portion = torch.tensor(list(ans_num_dict.values()), dtype=torch.float32)
        qt_dict_freq[qt] = (ans, portion)
        qt_cnt[qt] = qt_cnt.get(qt, 0) + torch.sum(portion)
        qt_ans_cnt[qt] = len(ans)
    # total = sum(qt_cnt.values())
    for qt in qt_scale:
        qt_scale[qt] = {int(k): v for k, v in qt_scale[qt].items()}
        # print('qt:', qt)
        for k, v in qt_scale[qt].items():
            qt_scale[qt][k] = float(1.0) / v / qt_cnt[qt]
            # print(qt_scale[qt][k])

    return qt_dict, qt_dict_freq, qt_scale


class VQAFeatureDataset(Dataset):
    def __init__(self, name, dictionary, dataset):
        super(VQAFeatureDataset, self).__init__()
        print('name:', name)
        if dataset == 'vqace':
            assert name in ['train', 'all', 'cou', 'easy', 'hard']
        else:
            assert name in ['train', 'val', 'test']
        self.split = name
        config.dataset = dataset
        self.dictionary = dictionary

        # loading answer-label
        self.ans2label = json.load(open(os.path.join(config.cache_root,
            'traintest_ans2label.json'), 'r'))
        print(os.path.join(config.cache_root,
            'traintest_ans2label.json'))

        self.label2ans = json.load(open(os.path.join(config.cache_root,
            'traintest_label2ans.json'), 'r'))
        self.num_ans_candidates = len(self.ans2label)
        print('name:', name)
        print('self.num_ans_candidates:', self.num_ans_candidates)

        # loading image features
        if name== 'test':
            self.img_id2idx = json.load(open(os.path.join(config.ids_path,
                                                        'test36_imgid2idx.json'), 'r'))
            print(os.path.join(config.ids_path,
                               'test36_imgid2idx.json'))
        else:
            if dataset == 'vqace':
                self.img_id2idx = json.load(open(os.path.join(
                    config.ids_path, '{}36_imgid2idx.json'.format(
                        name)), 'r'))
                print(os.path.join(config.ids_path,
                                    '{}36_imgid2idx.json'.format(name)))
            else:
                self.img_id2idx = json.load(open(os.path.join(config.ids_path,
                                                        'train36_imgid2idx.json'), 'r'))
                print(os.path.join(config.ids_path,
                                    'train36_imgid2idx.json'))

        self.entries = _load_dataset(config.cache_root, name, self.img_id2idx,ratio=1.0)
        self.margins, self.freq, self.qt_scale = _load_margin(config.cache_root, name)

        self.h5_path = os.path.join(config.rcnn_path, '{}_obj36.h5'.format(name))
        print(self.h5_path)

        # Convert list to dict (for evaluation)
        self.id2datum = {
            datum['question_id']: {'labels': datum['answer']['labels'], 'scores': datum['answer']['scores']}
            for datum in self.entries
        }

        self.tokenize()
        self.tensorize()
        self.v_dim = config.output_features
        self.s_dim = config.num_fixed_boxes

    def tokenize(self, max_length=config.max_question_len):
        """ Tokenizes the questions.
            This will add q_token in each entry of the dataset.
            -1 represent nil, and should be treated as padding_idx in embedding.
        """
        for entry in self.entries:
            tokens = self.dictionary.tokenize(entry['question'], False, True)
            tokens = tokens[:max_length]
            if len(tokens) < max_length:
                # Note here we pad in front of the sentence
                padding = [self.dictionary.padding_idx] * (max_length - len(tokens))
                tokens = padding + tokens
            utils.assert_eq(len(tokens), max_length)
            entry['q_token'] = tokens

    def tensorize(self):
        if config.in_memory:
            self.features = torch.from_numpy(self.features)

        for entry in self.entries:
            question = torch.from_numpy(np.array(entry['q_token']))
            entry['q_token'] = question

            answer = entry['answer']
            labels = np.array(answer['labels'])
            scores = np.array(answer['scores'], dtype=np.float32)
            if len(labels):
                labels = torch.from_numpy(labels)
                scores = torch.from_numpy(scores)
                entry['answer']['labels'] = labels
                entry['answer']['scores'] = scores
            else:
                entry['answer']['labels'] = None
                entry['answer']['scores'] = None

    def load_image(self, image_id):
        """ Load one image feature. """
        if not hasattr(self, 'image_feat'):
            self.image_feat = h5py.File(self.h5_path, 'r')
        features = self.image_feat['image_features'][image_id]
        boxes = self.image_feat['image_bb'][image_id]
        return features, boxes

    def __getitem__(self, index):
        entry = self.entries[index]
        
        question_id = entry['question_id']
        question = entry['q_token']
        answer = entry['answer']
        q_type = answer['question_type']
        ans_type = entry['answer_type']
        labels = answer['labels']
        scores = answer['scores']
        target = torch.zeros(self.num_ans_candidates)

        # Get image info
        features, _ = self.load_image(entry['image'])

        margin_label, margin_score = self.margins[q_type]
        freq_label, freq_score = self.freq[q_type]
        scale = torch.tensor([self.qt_scale[q_type][l.item()] for l in freq_label])
        scale = F.softplus(scale)

        betas = [0]
        torch.set_printoptions(profile="full")
        idx = 0
        eff = 1 - torch.float_power(betas[idx], freq_score)
        per0 = (1 - betas[idx]) / eff
        per0 = per0 / torch.sum(per0) * freq_score.shape[0]
        per0 = per0.float()

        target_margin = torch.zeros(self.num_ans_candidates)
        freq_margin0 = torch.zeros(self.num_ans_candidates)
        scale_margin = torch.zeros(self.num_ans_candidates)

        if labels is not None:
            target.scatter_(0, labels, scores)
            target_margin.scatter_(0, margin_label, margin_score)
            freq_margin0.scatter_(0, freq_label, per0)
            scale_margin.scatter_(0, freq_label, scale)
        return features, question, target, target_margin, scale_margin, question_id, freq_margin0, q_type, ans_type

    def __len__(self):
        return len(self.entries)
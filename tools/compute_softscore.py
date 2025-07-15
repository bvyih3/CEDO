import os
import sys
import json
sys.path.append(os.getcwd())

import numpy as np
from scipy.stats import entropy
from collections import Counter, defaultdict
import argparse
import utils.utils as utils
import utils.config as config
from main_arcface_multioptim import parse_args


def get_score(occurences):
    """ Average over all 10 choose 9 sets. """
    score_soft = occurences * 0.3
    score = score_soft if score_soft < 1.0 else 1.0
    return 1.0


def filter_answers(answers_dset, min_occurence):
    """ Filtering answers whose frequency is less than min_occurence. """
    occurence = {}
    for ans_entry in answers_dset:
        gtruth = str(ans_entry['answer'])
        # gtruth = utils.preprocess_answer(gtruth)
        if gtruth not in occurence:
            occurence[gtruth] = set()
        occurence[gtruth].add(ans_entry['question_id'])
    for answer in list(occurence.keys()):
        if len(occurence[answer]) < min_occurence:
            occurence.pop(answer)

    print('Num of answers that appear >= {} times: {}'.format(
                                min_occurence, len(occurence)))
    return occurence


def create_ans2label(occurence, name, cache_root):
    """ Map answers to label. """
    label, label2ans, ans2label = 0, [], {}
    for answer in occurence:
        label2ans.append(answer)
        ans2label[answer] = label
        label += 1

    utils.create_dir(cache_root)
    print('ans_num:', len(label2ans))
    cache_file = os.path.join(cache_root, name+'_ans2label.json')
    json.dump(ans2label, open(cache_file, 'w'))
    cache_file = os.path.join(cache_root, name+'_label2ans.json')
    json.dump(label2ans, open(cache_file, 'w'))
    return ans2label


def compute_target(answers_dset, ans2label, name, cache_root):
    """ Augment answers_dset with soft score as label. """
    target = []
    tot = 0
    for ans_entry in answers_dset:
        answer = str(ans_entry['answer'])
        answer_count = {}
        answer_count[answer] = answer_count.get(answer, 0) + 1

        labels, scores = [], []
        cnt = 0
        for answer in answer_count:
            if answer not in ans2label:
                continue
            cnt += 1
            labels.append(ans2label[answer])
            # score = get_score(answer_count[answer])
            

        if cnt == 0:
            tot += 1
            # scores.append(0.0)
            # print(answer)
            # print(labels)
            # print(scores)
        else:
            scores.append(1.0)

        if 'question_type' not in ans_entry:
            ans_entry['question_type'] = 'notype'
        if 'answer_type' not in ans_entry:
            ans_entry['answer_type'] = 'notype'

        target.append({
            'question_type': ans_entry['question_type'],
            'question_id': ans_entry['question_id'],
            'image_id': ans_entry['image_id'],
            'labels': labels,
            'scores': scores,
            'answer_type': ans_entry['answer_type']
        })
    print('丢弃答案数量:', tot)
    utils.create_dir(cache_root)
    cache_file = os.path.join(cache_root, name+'_target.json')
    print('data_len:', len(target))
    print('use num:', len(target) - tot)
    json.dump(target, open(cache_file, 'w'))


def extract_type(answers_dset, name, ans2label, cache_root):
    """ Extract answer distribution for each question type. """
    qt_dict = defaultdict(list)
    for ans_entry in answers_dset:
        qt = ans_entry['question_type']
        ans_idxs = []
        ans = str(ans_entry['answer'])
        # ans = utils.preprocess_answer(ans)
        ans_idx = ans2label.get(ans, None)
        if ans_idx is not None:
            ans_idxs.append(ans_idx)
        qt_dict[qt].extend(ans_idxs) # counting later

    number = 0
    # count answers for each question type
    for qt in qt_dict:
        ans_num_dict = Counter(qt_dict[qt])
        ans_num_dict = {k: v
            for k, v in ans_num_dict.items() if v >= 0}
        total_num = sum(ans_num_dict.values())
        for ans, ans_num in ans_num_dict.items():
            # ans_num_dict[ans] = float(total_num - ans_num) / ans_num
            ans_num_dict[ans] = float(ans_num) / total_num

        values = np.array(list(ans_num_dict.values()), dtype=np.float32)
        if entropy(values + 1e-6, base=2) >= config.entropy:
            qt_dict[qt] = {k: 0.0 for k in ans_num_dict}
            number += 1
        else:
            qt_dict[qt] = ans_num_dict
        # qt_dict[qt] = ans_num_dict
        # if qt == "PRES, COLOR":
        #     print(ans_num_dict)
    cache_file = os.path.join(cache_root, name + '_margin.json')
    json.dump(qt_dict, open(cache_file, 'w'))
    qt_dict = defaultdict(list)
    for ans_entry in answers_dset:
        qt = ans_entry['question_type']
        ans_idxs = []
        ans = str(ans_entry['answer'])
        # ans = utils.preprocess_answer(ans)
        ans_idx = ans2label.get(ans, None)
        if ans_idx is not None:
            ans_idxs.append(ans_idx)
        qt_dict[qt].extend(ans_idxs)  # counting later


    for qt in qt_dict:
        ans_num_dict = Counter(qt_dict[qt])
        ans_num_dict = {k: v
            for k, v in ans_num_dict.items() if v >= 0}

        qt_dict[qt] = ans_num_dict
    cache_file = os.path.join(cache_root, name + '_freq.json')
    json.dump(qt_dict, open(cache_file, 'w'))

if __name__ == '__main__':
    args = parse_args()
    print(args)
    dataset = args.dataset
    config.dataset = dataset
    config.update_paths(args.dataset)

    train_answers = utils.get_file(train=True, answer=True)
    test_answers = utils.get_file(test=True, answer=True)

    answers = train_answers + test_answers
    print("filtering answers less than minimum occurrence...")
    config.min_occurence = 9
    occurence = filter_answers(answers, config.min_occurence)
    print("create answers to integer labels...")
    ans2label = create_ans2label(occurence, 'traintest', config.cache_root)

    print("converting target for train and test answers...")
    compute_target(train_answers, ans2label, 'train', config.cache_root)
    compute_target(test_answers, ans2label, 'test', config.cache_root)

    print("extracting answer margin for each question type...")
    extract_type(train_answers, 'train', ans2label, config.cache_root)
    extract_type(test_answers, 'test', ans2label, config.cache_root)

import os
import sys
import csv
import h5py
import json
import base64
import argparse
from tqdm import tqdm
import numpy as np
sys.path.append(os.getcwd())
csv.field_size_limit(sys.maxsize)

import utils.utils as utils
import utils.config as config


FIELDNAMES = ["img_id", "img_h", "img_w", "objects_id", "objects_conf",
              "attrs_id", "attrs_conf", "num_boxes", "boxes", "features"]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=['train', 'test'], default='train')
    parser.add_argument(
        '--dataset', default='slake',
        choices=["slake", "slake-cp", "vqa-rad", "vqa-rad-cp", "omni", "omni-cp", "vqacp-v2", "vqace", "gqaood", "pmc", "pmc-cp", "pmc2", "pmc2-cp", "pmca", "pmca-cp"],
        help='choose dataset'
    )
    args = parser.parse_args()
    dataset = args.dataset
    config.dataset = dataset
    config.update_paths(dataset)

    print('config.dataset:', config.dataset)

    split_set = ['train'] if args.split == 'train' else ['test']
    # if dataset == 'vqace' and args.split == 'test':
    #     split_set = ['trainval']

    # choose the right bottom up feature file
    bottom_up_path = os.path.join(config.bottom_up_path, dataset + '_obj36.tsv')
    print(bottom_up_path)
    
    num_images = config.test_num_images if args.split == 'test' \
                else config.trainval_num_images
    print('num_images:', num_images)


    # load all image ids
    img_ids = []
    for split in split_set:
        split_ids_path = os.path.join(config.ids_path, split + '_ids.json')
        print(split_ids_path)
        if os.path.exists(split_ids_path):
            img_ids += json.load(open(split_ids_path, 'r'))
        else:
            print('not exist!')
            exit(0)
    print('len(img_ids):', len(img_ids))

    errorid = []

    print("Start to load Faster-RCNN detected objects from %s" % bottom_up_path)
    with open(bottom_up_path, 'r') as tsv_in_file:
        reader = csv.DictReader(
                tsv_in_file, delimiter='\t', fieldnames=FIELDNAMES)
        for item in tqdm(reader, total=num_images):
            image_id = item['img_id']
            if dataset.startswith('slake') or dataset.startswith("pmc"):
                image_id = int(image_id)
            item['num_boxes'] = int(item['num_boxes'])
            if item['num_boxes'] < 36:
                errorid.append(image_id)
    print(len(errorid))
    json.dump(errorid, open('errorid.json', 'w'))
                # if item['boxes'].startswith("b'") and item['boxes'].endswith("'"):
                #     item['boxes'] = item['boxes'][2:][:-1]
                # buf = base64.b64decode(item['boxes'])
                # bboxes = np.frombuffer(buf,
                #     dtype=np.float32).reshape((item['num_boxes'], 4))



    print("done!")


if __name__ == '__main__':
    main()

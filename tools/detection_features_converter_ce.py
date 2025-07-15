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
    parser.add_argument("--split", choices=['train', 'all', 'cou', 'easy', 'hard'], default='train')
    parser.add_argument(
        '--dataset', default='vqace',
        choices=["vqace"],
        help='choose dataset'
    )
    args = parser.parse_args()
    dataset = args.dataset
    config.dataset = dataset
    config.update_paths(dataset)

    print('config.dataset:', config.dataset)
    split_set = [args.split]

    # choose the right bottom up feature file
    bottom_up_path = os.path.join(config.bottom_up_path, dataset + '_obj36.tsv')
    print(bottom_up_path)


    # dump indices
    split_indices_path = os.path.join(
        './data/vqace/', args.split + '36_imgid2idx.json')

    # load all image ids
    img_ids = []
    for split in split_set:
        split_ids_path = os.path.join('./data/vqace/', split + '_ids.json')
        print(split_ids_path)
        if os.path.exists(split_ids_path):
            img_ids += json.load(open(split_ids_path, 'r'))
        else:
            split_year = '2014' if not split == 'test' else '2015'
            split_image_path = os.path.join(
                config.image_path, split + split_year)
            img_ids_dump = utils.load_imageid(split_image_path)
            json.dump(list(img_ids_dump), open(split_ids_path, 'w'))
            img_ids += json.load(open(split_ids_path, 'r'))

    num_images = len(img_ids)
    # create h5 files
    h_split = h5py.File(os.path.join(
            config.rcnn_path, args.split + '_obj36.h5'), 'w')
    split_img_features = h_split.create_dataset(
        'image_features', (len(img_ids),
        config.num_fixed_boxes, config.output_features), 'f')
    split_img_bb = h_split.create_dataset(
        'image_bb', (len(img_ids),
        config.num_fixed_boxes, 4), 'f')
    # split_img_w = h_split.create_dataset(
    #     'image_w', (len(img_ids), 1), 'f')
    # split_img_h = h_split.create_dataset(
    #     'image_h', (len(img_ids), 1), 'f')

    counter, indices = 0, {}
    print("Start to load Faster-RCNN detected objects from %s" % bottom_up_path)
    with open(bottom_up_path, 'r') as tsv_in_file:
        reader = csv.DictReader(
                tsv_in_file, delimiter='\t', fieldnames=FIELDNAMES)
        for item in tqdm(reader, total=num_images):
            image_id = item['img_id']
            if dataset == 'vqacp-v2' or dataset == 'vqace':
                image_id = int(image_id.split('_')[2].lstrip('0'))
            if dataset == 'vqace':
                if args.split == 'train':
                    image_id = str(image_id)
                else:
                    image_id = int(image_id)
            if image_id in img_ids:
                item['num_boxes'] = int(item['num_boxes'])
                image_w = int(item['img_w'])
                image_h = int(item['img_h'])
                if item['boxes'].startswith("b'") and item['boxes'].endswith("'"):
                    item['boxes'] = item['boxes'][2:][:-1]
                buf = base64.b64decode(item['boxes'])
                bboxes = np.frombuffer(buf,
                    dtype=np.float32).reshape((item['num_boxes'], 4))
                
                bboxes_copy = bboxes.copy()
                bboxes_copy[:, (0, 2)] /= image_w
                bboxes_copy[:, (1, 3)] /= image_h

                img_ids.remove(image_id)
                indices[image_id] = counter
                split_img_bb[counter, :, :] = bboxes_copy
                if item['features'].startswith("b'") and item['features'].endswith("'"):
                    item['features'] = item['features'][2:][:-1]
                buf = base64.b64decode(item['features'])
                split_img_features[counter, :, :] = np.frombuffer(buf,
                    dtype=np.float32).reshape((item['num_boxes'], -1))
                if int(image_h) == 0:
                    print('image_w:', image_w)
                    print('image_h:', image_h)
                    print('bboxes:', split_img_bb[counter, :, :])
                    print('features:', split_img_features[counter, :, :])
                counter += 1
            if len(img_ids) == 0:
                break

    if len(img_ids) != 0:
        print("Warning: {}_image_ids is not empty".format(args.split))

    print("done!")
    json.dump(indices, open(split_indices_path, 'w'))
    h_split.close()


if __name__ == '__main__':
    main()

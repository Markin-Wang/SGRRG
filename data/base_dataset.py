import os
import json
import torch
from PIL import Image
from torch.utils.data import Dataset
import multiprocessing

import torchvision.transforms as tfs
import cv2
import pandas as pd
from modules.utils import GetTransforms, transform
import numpy as np
import pyarrow as pa
import io
from modules.tokenizers_origin import Tokenizer
import torchvision
from collections import defaultdict

torchvision.disable_beta_transforms_warning()
from torchvision import datapoints
from torchvision.transforms.v2 import functional as F
import torchvision.transforms.v2 as transforms

class BaseDatasetArrow(Dataset):
    def __init__(self, config, split, tokenizer, transform=None, text_column_name='caption', name=None, test=None):
        self.max_seq_length = config['max_seq_length']
        self.split = split
        self.tokenizer = tokenizer
        self.transform = transform
        self.test = test
        self.debug = config['debug']
        self.dataset_name = config['dataset_name']
        self.att_cls = config['att_cls']
        self.dsr = config['dsr']  # down sample rate
        root = os.path.join(config['data_dir'], config['dataset_name'])
        self.table = pa.ipc.RecordBatchFileReader(pa.memory_map(f"{root}/{name}.arrow", "r")).read_all()
        if self.dataset_name == 'mimic_cxr':
            # if self.split != 'train':
            #     img_filter_path = os.path.join(root, 'annotations', f"{name}.json")
            #     with open(img_filter_path, 'r') as f:
            #         img_filter_dict = json.load(f)
            #     img_keys = set([img['id'] for img in img_filter_dict['images']])
            #     print(img_filter_dict.keys())
            #     mask = [self.table['image_id'][i].as_py() in img_keys for i in range(len(self.table['image_id']))]
            #     print('before:', len(self.table['image_id']))
            #     self.table = self.table.filter(mask)
            #     with pa.OSFile(os.path.join(root, f'mimic_cxr_{split}_filter.arrow'), "wb") as sink:
            #         with pa.RecordBatchFileWriter(sink, self.table.schema) as writer:
            #             writer.write_table(self.table)
            #     print('after:', len(self.table['image_id']))
            self.attributes_path = os.path.join(root, 'annotations', 'attribute_anns_id.json')
            self.attributes = json.loads(open(self.attributes_path, 'r').read())
            self.att_labels = self.attributes['annotations']
            self.att_cat_info = self.attributes['category_info']
        self.text_column_name = text_column_name
        # self.all_texts = self.table[text_column_name].to_pandas().tolist()
        self.all_texts = self.table[text_column_name].to_pandas()

        if split == 'train':
            self.tokenizer = Tokenizer(config, self.all_texts)

        if self.att_cls:
            if split == 'train':
                ann_file_path = os.path.join(root, 'annotations', 'mimic_cxr_train.json')
            else:
                ann_file_path = os.path.join(root, 'annotations', f'mimic_cxr_{split}_dino.json')
            self.data_infos = self.load_box_annotations(ann_file_path)

        self.labels_path = os.path.join(root, config['label_path'])
        self.labels = json.loads(open(self.labels_path, 'r').read())

        self.to_tensor_trans = transforms.Compose([
                transforms.ToImageTensor(),  # # note this does not scale the image
                transforms.ConvertImageDtype(torch.float32),
                transforms.Normalize((123.675, 116.28, 103.53),
                                     (58.395, 57.12, 57.375))
            ])

    def get_raw_image(self, index, image_key="image"):
        image_bytes = io.BytesIO(self.table[image_key][index].as_py())
        image_bytes.seek(0)
        return Image.open(image_bytes).convert("RGB")

    def get_image(self, index, image_key="image"):
        iid = self.table['image_id'][index].as_py()
        if 'iu_xray' in self.dataset_name:
            array = iid.split('-')
            modified_id = array[0] + '-' + array[1]
            label = np.array(self.labels[modified_id]).astype(np.float32)
        else:
            label = np.array(self.labels[iid]).astype(np.float32)
        if 'iu_xray' in self.dataset_name:
            image1 = self.get_raw_image(index, image_key='image1')
            image2 = self.get_raw_image(index, image_key='image2')
            image_tensor1 = [self.transform['norm_to_tensor'](self.transform['common_aug'](image1))]
            image_tensor2 = [self.transform['norm_to_tensor'](self.transform['common_aug'](image2))]
            image_tensor = torch.stack(image_tensor1 + image_tensor2, dim=0)
        else:
            image = self.get_raw_image(index, image_key=image_key)
            if self.att_cls:
                box_ann = self.get_box(iid)
                bboxes = datapoints.BoundingBox(box_ann['bboxes'], format=datapoints.BoundingBoxFormat.XYXY,
                                                spatial_size=box_ann['spatial_size']
                                                )
                image_tensor, bboxes = self.transform['common_aug'](image, {"boxes": bboxes, "labels": box_ann['labels']})
                image_tensor =  self.transform['norm_to_tensor'](image_tensor)
            else:
                image_tensor = self.transform['common_aug'](image)
                image_tensor = self.transform['norm_to_tensor'](image_tensor)
        return_dict = {
            "image": image_tensor,
            "img_id": iid,
            "img_index": index,
            "raw_index": index,
            'img_labels': label,
        }
        if self.att_cls:
            return_dict.update({'bboxes':bboxes,'bboxes_label':box_ann['labels']})
        # image = self.get_raw_image(index, image_key=image_key)
        # image_tensor = [tr(image) for tr in self.transforms]
        return return_dict

    def get_box(self, image_id):
        ann_ids = self.id2anns[image_id]
        ann_info = self.annotations[ann_ids]
        img_info = self.data_infos[image_id]
        return self._parse_ann_info(img_info, ann_info)

    def load_box_annotations(self, ann_file):
        """Load annotation from COCO style annotation file.
        Args:
            ann_file (str): Path of annotation file.
        Returns:
            list[dict]: Annotation info from COCO api.
        """
        with open(ann_file, 'r') as f:
            self.mimic_cxr = json.load(f)
        self.id2anns = {}
        self.annotations = np.array(self.mimic_cxr['annotations'])
        self.catToImgs = defaultdict(list)
        for ann in self.annotations:
            self.catToImgs[ann['category_id']].append(ann['image_id'])
        # The order of returned `cat_ids` will not
        # change with the order of the CLASSES
        self.cat_ids = [cat_info['id'] for cat_info in self.mimic_cxr['categories']]
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        data_infos = {}
        total_ann_ids = []
        for img_info in self.mimic_cxr['images']:
            img_info['filename'] = img_info['file_name']
            ann_ids = img_info['ann_ids']
            data_infos[img_info['id']] = img_info
            self.id2anns[img_info['id']] = ann_ids
            total_ann_ids.extend(ann_ids)
        assert len(set(total_ann_ids)) == len(
            total_ann_ids), f"Annotation ids in '{ann_file}' are not unique!"
        return data_infos

    def _parse_ann_info(self, img_info, ann_info):
        """Parse bbox and mask annotation.
        Args:
            ann_info (list[dict]): Annotation info of an image.
            with_mask (bool): Whether to parse mask annotations.
        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,\
                labels, masks, seg_map. "masks" are raw annotations and not \
                decoded into binary masks.
        """
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_masks_ann = []
        groundtruth_is_crowd = []
        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            if ann.get('iscrowd', False) and (
                    not self.iscrowd):  # while training, skip iscrowd
                continue

            x1, y1, w, h = ann['bbox']
            x1, y1, w, h = x1 / self.dsr, y1 / self.dsr, w / self.dsr, h / self.dsr
            inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            if ann['category_id'] not in self.cat_ids:
                continue

            bbox = [x1, y1, x1 + w, y1 + h]

            if ann.get('iscrowd', False):
                gt_bboxes.append(
                    bbox
                )  # add crowded gt bboxes when eval, but not needed in training
                gt_labels.append(self.cat2label[ann['category_id']])
                gt_masks_ann.append(ann.get('segmentation', None))
                gt_bboxes_ignore.append(bbox)
                groundtruth_is_crowd.append(1)
            else:
                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2label[ann['category_id']])
                gt_masks_ann.append(ann.get('segmentation', None))
                groundtruth_is_crowd.append(0)

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
            groundtruth_is_crowd = np.array(
                groundtruth_is_crowd, dtype=np.int8)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)
            groundtruth_is_crowd = np.array([], dtype=np.int8)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        seg_map = img_info['filename'].replace('jpg', 'png')

        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            spatial_size=(h,w),
            groundtruth_is_crowd=groundtruth_is_crowd,
            bboxes_ignore=gt_bboxes_ignore,
            masks=gt_masks_ann,
            seg_map=seg_map)

        return ann

    def xyxy2xywh(self, bbox):
        """Convert ``xyxy`` style bounding boxes to ``xywh`` style for COCO
        evaluation.
        Args:
            bbox (numpy.ndarray): The bounding boxes, shape (4, ), in
                ``xyxy`` order.
        Returns:
            list[float]: The converted bounding boxes, in ``xywh`` order.
        """

        _bbox = bbox.tolist()
        return [
            _bbox[0],
            _bbox[1],
            _bbox[2] - _bbox[0],
            _bbox[3] - _bbox[1],
        ]

    def get_text(self, index):
        text = self.all_texts[index][0]  # only one gt caption in rrg
        encoding = self.tokenizer(text)[:self.max_seq_length]
        mask = [1] * len(encoding)
        gt_text = self.all_texts[index]
        seq_length = len(encoding)
        return {
            "text": encoding,
            "img_index": index,
            "cap_index": index,
            "mask": mask,
            "raw_index": index,
            "gt_txt": gt_text,
            "seq_length": seq_length,
        }

    def get_suite(self, index):
        ret = dict()
        ret.update(self.get_image(index))
        txt = self.get_text(index)
        ret.update(txt)
        return ret

    def __len__(self):
        return len(self.all_texts) if not self.debug else len(self.all_texts) // 100





class BaseDataset(Dataset):
    def __init__(self, config, tokenizer, split, transform=None, test=None):
        self.max_seq_length = config['max_seq_length']
        self.split = split
        self.tokenizer = tokenizer
        self.transform = transform
        self.test = test
        root = os.path.join(config['data_dir'], config['dataset_name'])
        self.image_dir = os.path.join(root, 'images')
        self.ann_path = os.path.join(root, 'annotation.json')
        self.ann = json.loads(open(self.ann_path, 'r').read())
        self.labels_path = os.path.join(root, config['label_path'])
        self.labels = json.loads(open(self.labels_path, 'r').read())
        self.dataset_name = config['dataset_name']
        if self.test:
            selected_parts = ['p10']
        self.examples = self.ann[self.split]
        if self.test and self.split == 'train' and self.dataset_name != 'iu_xray':
            self.examples = [e for part in selected_parts for e in self.examples if part == e['image_path'][0][:3]]
        if self.dataset_name == 'iu_xray':
            self._labels = []
            for e in self.examples:
                img_id = e['id']
                array = img_id.split('-')
                modified_id = array[0] + '-' + array[1]
                self._labels.append(self.labels[modified_id])
        else:
            self._labels = [self.labels[e['id']] for e in self.examples]

        for i in range(len(self.examples)):
            self.examples[i]['ids'] = tokenizer(self.examples[i]['report'])
            self.examples[i]['mask'] = [1] * len(self.examples[i]['ids'])
        # print(111111, len(self.examples))

    def __len__(self):
        return len(self.examples)

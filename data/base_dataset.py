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
from tqdm import tqdm
import torch.distributed as dist
from config import cgnome_catid2attrange as catid2attrange, categories, cgnome_id2cat as id2cat


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
        self.region_cls = config['region_cls']
        self.dsr = config['dsr']  # down sample rate
        root = os.path.join(config['data_dir'], config['dataset_name'])
        self.table = pa.ipc.RecordBatchFileReader(pa.memory_map(f"{root}/{name}.arrow", "r")).read_all()
        self.num_attributes = config['num_attributes']
        self.name2label = {name: i for i, name in enumerate(categories)}  # ensure the label setting consistency
        self.max_att = max(id2cat)

        self.text_column_name = text_column_name
        # self.all_texts = self.table[text_column_name].to_pandas().tolist()
        self.all_texts = self.table[text_column_name].to_pandas()
        self.no_region_count = 0

        if split == 'train':
            self.tokenizer = Tokenizer(config, self.all_texts)

        if self.dataset_name != 'iu_xray' and self.region_cls:
            # if self.split == 'train':
            # 159434 training images both in chest vg mimic-cxr training set
            # 113922 before in cgnome training set after 113480
            # 7 images without a scene graph given in training set 113473 finally
            # 2 images in val without sg
            # 3 images in test without sg

            # img_filter_path = os.path.join(root, 'annotations', f"{name}.json")
            # with open(img_filter_path, 'r') as f:
            #     img_filter_dict = json.load(f)
            # img_keys = set([img['id'] for img in img_filter_dict['images']])
            # print(img_filter_dict.keys())
            # filter training images based on no attributes get 158794 images
            # if dist.get_rank() == 0:
            #     no_attribute_ids = set(json.load(open(os.path.join(root, 'annotations', "no_sg_ids.json"), 'r'))[split])
            #     mask = [self.table['image_id'][i].as_py() not in no_attribute_ids for i in range(len(self.table['image_id']))]
            #     print('before:', len(self.table['image_id']))
            #     self.table = self.table.filter(mask)
            #     with pa.OSFile(os.path.join(root, f'cxr_gnome_{split}_ft_sg.arrow'), "wb") as sink:
            #         with pa.RecordBatchFileWriter(sink, self.table.schema) as writer:
            #             writer.write_table(self.table)
            #     print('after:', len(self.table['image_id']))
            #     exit()
            # Form box annotation

            if split == 'train':
                ann_file_path = os.path.join(root, 'annotations', f'box_train.json')
            else:
                ann_file_path = os.path.join(root, 'annotations', f'box_{split}_dino_th05.json')
                # ann_file_path = os.path.join(root, 'annotations', f'box_{split}.json')
            self.box_infos = self.load_box_annotations(ann_file_path)

            self.attributes_path = os.path.join(root, 'annotations', 'attribute_anns_id_mhead.json')
            self.attributes = json.loads(open(self.attributes_path, 'r').read())
            self.attribute_anns, self.region_anns = self._parse_att_ann_info()
            self.att_labels = self.attributes['annotations']
            self.att_cat_info = self.attributes['attribute_info']

    def get_raw_image(self, index, image_key="image"):
        image_bytes = io.BytesIO(self.table[image_key][index].as_py())
        image_bytes.seek(0)
        return Image.open(image_bytes).convert("RGB")

    def get_image(self, index, image_key="image"):
        iid = self.table['image_id'][index].as_py()
        if 'iu_xray' in self.dataset_name:
            image1 = self.get_raw_image(index, image_key='image1')
            image2 = self.get_raw_image(index, image_key='image2')
            image_tensor1 = [self.transform['norm_to_tensor'](self.transform['common_aug'](image1))]
            image_tensor2 = [self.transform['norm_to_tensor'](self.transform['common_aug'](image2))]
            image_tensor = torch.stack(image_tensor1 + image_tensor2, dim=0)
        else:
            image = self.get_raw_image(index, image_key=image_key)
            if self.region_cls:
                box_ann = self.get_box(iid)
                bboxes = datapoints.BoundingBox(box_ann['bboxes'], format=datapoints.BoundingBoxFormat.XYXY,
                                                spatial_size=box_ann['spatial_size']
                                                )
                image_tensor, box_ann = self.transform['common_aug'](image,
                                                                     {"boxes": bboxes, "labels": box_ann['labels']})

                image_tensor = self.transform['norm_to_tensor'](image_tensor)

                region_labels = self.get_region_label(image_id=iid)
                box_ann['box_masks'] = self.get_box_mask(box_ann['labels'], region_labels)
            if self.att_cls:
                attribute_labels = self.get_attribute_label(image_id=iid)

                # box masks are used to determine which mask will be selected
                # to perform scene graph embedding and attribute prediction
            else:
                image_tensor = self.transform['common_aug'](image)
                image_tensor = self.transform['norm_to_tensor'](image_tensor)
        return_dict = {
            "image": image_tensor,
            "img_id": iid,
            "img_index": index,
            "raw_index": index,
        }
        if self.region_cls:
            box_ann['box_labels'] = box_ann.pop('labels')
            return_dict.update(box_ann)
            return_dict.update({'region_labels': region_labels})

        if self.att_cls:
            name = "attribute_labels" if self.split == 'train' else "attribute_label_dicts"
            # name = "attribute_labels"
            return_dict.update({name: attribute_labels})

        return return_dict

    def get_box(self, image_id):
        ann_ids = self.id2anns[image_id]
        ann_info = self.box_annotations[ann_ids]
        img_info = self.box_infos[image_id]
        # print(11111,self._parse_box_ann_info(img_info, ann_info))
        return self._parse_box_ann_info(img_info, ann_info)

    def get_region_label(self, image_id):
        # some images without any regions mentioned, assign all zeros first,
        # 614 images in training filter set no attributes
        if image_id not in self.region_anns:
            self.no_region_count += 1
            print(f'{self.no_region_count} have no regions.')

        return self.region_anns.get(image_id, torch.zeros(1, len(categories)))

    def get_box_mask(self, box_labels, region_labels):
        box_masks = region_labels[0, box_labels] == 1
        return box_masks

    def get_attribute_label(self, image_id):
        # some images without any regions mentioned, empty first
        # may consider delete these images
        # 614 images in training filter set no attributes
        return self.attribute_anns.get(image_id, [])

    def get_attribute_label_multi(self, image_id):
        # some images without any regions mentioned, empty first
        # may consider delete these images
        # 614 images in training filter set no attributes
        return self.attribute_anns.get(image_id, [])

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

    def _test_att(self, img_id):
        att_labels = self.get_attribute_label(img_id)
        print(att_labels)
        exit()

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
        self.box_annotations = np.array(self.mimic_cxr['annotations'])
        self.catToImgs = defaultdict(list)
        for ann in self.box_annotations:
            self.catToImgs[ann['category_id']].append(ann['image_id'])
        # The order of returned `cat_ids` will not
        # change with the order of the CLASSES
        self.cat_ids = [cat_info['id'] for cat_info in self.mimic_cxr['categories']]
        self.cat2label = {cat_id: i for i, cat_id in
                          enumerate(self.cat_ids)}  # ensure label consistency for all processing
        box_infos = {}
        total_ann_ids = []
        for img_info in self.mimic_cxr['images']:
            img_info['filename'] = img_info['file_name']
            ann_ids = img_info['ann_ids']
            box_infos[img_info['id']] = img_info
            self.id2anns[img_info['id']] = ann_ids
            total_ann_ids.extend(ann_ids)
        assert len(set(total_ann_ids)) == len(
            total_ann_ids), f"Annotation ids in '{ann_file}' are not unique!"
        return box_infos

    def _parse_box_ann_info(self, img_info, ann_info):
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
        original_height, original_width = img_info['height'], img_info['width']
        img_height, img_width = original_height / self.dsr, original_width / self.dsr
        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            if ann.get('iscrowd', False) and (
                    not self.iscrowd):  # while training, skip iscrowd
                continue

            x1, y1, w, h = ann['bbox']
            x1, y1, w, h = x1 / self.dsr, y1 / self.dsr, w / self.dsr, h / self.dsr

            # dsr_h = img_height / 224
            # dsr_w = img_width / 224
            # if (w/dsr_w) // 32 < 2 or (h/dsr_h) // 32 < 2:
            #     self.count += 1
            #     print(self.count,(w/dsr_w) // 32,(h/dsr_h) // 32)
            # else:
            #     print('okay')

            inter_w = max(0, min(x1 + w, img_width) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_height) - max(y1, 0))
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
            spatial_size=(img_height, img_width),
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

    def _parse_att_ann_info(self):
        anns = self.attributes['annotations'][self.split]
        attribute_anns = {}
        region_anns = {}
        num_regions = len(categories)
        for k, v in anns.items():
            region_label = v['region_label']
            region_label_ = torch.zeros(1, num_regions)
            region_label_[0, region_label] = 1  # ensure the label consistency
            region_anns[k] = region_label_
            v.pop('region_label')
            attribute_anns[k] = {self.name2label[kk]: vv for kk, vv in v.items()}
            # transform to muli-hot vetor in collate_fn as too many classes, otherwise occpuy a large RAM
        return attribute_anns, region_anns

    def __len__(self):
        ratio = 20 if self.split == 'train' else 10
        return len(self.all_texts) // ratio if self.debug else len(self.all_texts)

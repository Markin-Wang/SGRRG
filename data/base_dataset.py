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
from config import id2cat

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
        self.name2label = {name:i for i,(name,_) in enumerate(id2cat)} # ensure the label setting consistency

        self.text_column_name = text_column_name
        # self.all_texts = self.table[text_column_name].to_pandas().tolist()
        self.all_texts = self.table[text_column_name].to_pandas()

        if split == 'train':
            self.tokenizer = Tokenizer(config, self.all_texts)


        if self.dataset_name == 'mimic_cxr' and self.att_cls:
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
            # Form box annotation
            if split == 'train':
                ann_file_path = os.path.join(root, 'annotations', 'mimic_cxr_train.json')
            else:
                ann_file_path = os.path.join(root, 'annotations', f'mimic_cxr_{split}_dino.json')
            self.data_infos = self.load_box_annotations(ann_file_path)


            self.attributes_path = os.path.join(root, 'annotations', 'attribute_anns_id.json')
            self.attributes = json.loads(open(self.attributes_path, 'r').read())
            self.attribute_anns, self.region_anns = self._parse_att_ann_info()
            self.att_labels = self.attributes['annotations']
            self.att_cat_info = self.attributes['category_info']



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
            if self.att_cls:
                box_ann = self.get_box(iid)
                bboxes = datapoints.BoundingBox(box_ann['bboxes'], format=datapoints.BoundingBoxFormat.XYXY,
                                                spatial_size=box_ann['spatial_size']
                                                )
                image_tensor, box_ann = self.transform['common_aug'](image, {"boxes": bboxes, "labels": box_ann['labels']})
                image_tensor =  self.transform['norm_to_tensor'](image_tensor)
                region_labels = self.get_region_label(image_id=iid)
                attribute_labels = self.get_attribute_label(image_id=iid)
            else:
                image_tensor = self.transform['common_aug'](image)
                image_tensor = self.transform['norm_to_tensor'](image_tensor)
        return_dict = {
            "image": image_tensor,
            "img_id": iid,
            "img_index": index,
            "raw_index": index,
        }
        if self.att_cls:
            box_ann['box_labels'] = box_ann.pop('labels') - 1 # the category id for mimic cxr strats from 1
            return_dict.update(box_ann)
            return_dict.update({'region_labels':region_labels,"attribute_labels":attribute_labels})

        return return_dict

    def get_box(self, image_id):
        ann_ids = self.id2anns[image_id]
        ann_info = self.annotations[ann_ids]
        img_info = self.data_infos[image_id]
        return self._parse_box_ann_info(img_info, ann_info)

    def get_region_label(self,image_id):
        # some images without any regions mentioned, assign all zeros first,
        # 614 images in training filter set no attributes
        return self.region_anns.get(image_id,torch.zeros(1, len(id2cat)))

    def get_attribute_label(self,image_id):
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

    def __len__(self):
        return len(self.all_texts) // 1000  if self.debug and self.split=='train' else len(self.all_texts)



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
            spatial_size=(img_height,img_width),
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
        attribute_anns = self.attributes['annotations']
        region_anns = {}
        for k,v in attribute_anns.items():
            categories = [self.name2label[category] for category in v.keys()] # ensure the label consistency
            region_label = torch.zeros(1, len(id2cat))
            region_label[0, categories] = 1.0
            region_anns[k] = region_label
        return attribute_anns, region_anns



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

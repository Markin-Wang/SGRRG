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
torchvision.disable_beta_transforms_warning()
from torchvision import datapoints

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
        self.dsr = config['dsr'] # down sample rate
        root = os.path.join(config['data_dir'], config['dataset_name'])
        self.table = pa.ipc.RecordBatchFileReader(pa.memory_map(f"{root}/{name}.arrow", "r")).read_all()
        if self.dataset_name == 'mimic_cxr':
            img_filter_path = os.path.join(root, 'annotations', f"{name}.json")
            with open(img_filter_path,'r') as f:
                img_filter_dict = json.load(f)
            img_keys =set([img['id'] for img in img_filter_dict['images']])
            print(img_filter_dict.keys())
            mask = [self.table['image_id'][i].as_py() in img_keys for i in range(len(self.table['image_id']))]
            print('before:', len(self.table['image_id']))
            self.table = self.table.filter(mask)
            print('after:',len(self.table['image_id']))
            self.attributes_path = os.path.join(root, 'annotations', 'attribute_anns_id.json')
            self.attributes = json.loads(open(self.attributes_path, 'r').read())
            self.att_labels = self.attributes['annotations']
            self.att_cat_info = self.attributes['category_info']
        self.text_column_name = text_column_name
        # self.all_texts = self.table[text_column_name].to_pandas().tolist()
        self.all_texts = self.table[text_column_name].to_pandas()

        if split == 'train':
            self.tokenizer = Tokenizer(config,self.all_texts)

        if self.att_cls:
            if split == 'train':
                ann_file_path = os.path.join(root,'annotations','mimic_cxr_train.json')
            else:
                ann_file_path = os.path.join(root, 'annotations', f'mimic_cxr_{split}_dino.json')
            self.data_infos = self.load_box_annotations(ann_file_path)

        self.labels_path = os.path.join(root, config['label_path'])
        self.labels = json.loads(open(self.labels_path, 'r').read())


    def get_raw_image(self, index, image_key="image"):
        image_bytes = io.BytesIO(self.table[image_key][index].as_py())
        image_bytes.seek(0)
        return Image.open(image_bytes).convert("RGB")

    def get_image(self, index, image_key="image"):
        if 'iu_xray' in self.dataset_name:
            image1 = self.get_raw_image(index, image_key='image1')
            image2 = self.get_raw_image(index, image_key='image2')
            image_tensor1 = [self.transform(image1)]
            image_tensor2 = [self.transform(image2)]
            image_tensor = torch.stack(image_tensor1 + image_tensor2,dim=0)
        else:
            image = self.get_raw_image(index, image_key=image_key)
            image = datapoints.Image(image)
            image_tensor = self.transform(image)
        # image = self.get_raw_image(index, image_key=image_key)
        # image_tensor = [tr(image) for tr in self.transforms]
        iid = self.table['image_id'][index].as_py()
        if 'iu_xray' in self.dataset_name:
            array = iid.split('-')
            modified_id = array[0] + '-' + array[1]
            label = np.array(self.labels[modified_id]).astype(np.float32)
        else:
            label = np.array(self.labels[iid]).astype(np.float32)
        return {
            "image": image_tensor,
            "img_id": iid,
            "img_index": index,
            "raw_index": index,
            'img_labels': label,
        }

    def get_box(self,image_id):
        ann_ids = self.id2anns[image_id]
        ann_info = self.annotations[ann_ids]
        img_info = self.data_infos[image_id]
        return self._parse_ann_info(img_info,ann_info)

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
            data_infos[ann_ids] = img_info
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
            x1, y1, w, h = x1/self.dsr, y1/self.dsr, w/self.dsr, h/self.dsr
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
        text = self.all_texts[index][0] # only one gt caption in rrg
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
        return len(self.all_texts) if not self.debug else len(self.all_texts)//100


class IuxrayMultiImageDatasetArrow(BaseDatasetArrow):
    def __init__(self, *args, split="", **kwargs):
        assert split in ["train", "val", "test"]
        self.split = split

        if split == "train":
            name = "iu_xray_train"
        elif split == "val":
            name = "iu_xray_val"
        elif split == "test":
            name = "iu_xray_test"
        else:
            name = None

        super().__init__(*args, **kwargs, split=split,name=name, text_column_name="caption")

    def __getitem__(self, index):
        suite = self.get_suite(index)

        if "test" in self.split or 'val' in self.split:
            iid = self.table["image_id"][index].as_py()
            # iid = int(iid.split(".")[0].split("_")[-1])
            suite.update({"iid": iid})

        return suite

class MIMICMultiImageDatasetArrow(BaseDatasetArrow):
    def __init__(self, *args, split="", **kwargs):
        assert split in ["train", "val", "test"]
        self.split = split

        if split == "train":
            name = "mimic_cxr_train"
        elif split == "val":
            name = "mimic_cxr_val"
        elif split == "test":
            name = "mimic_cxr_test"
        else:
            name = None

        super().__init__(*args, **kwargs, split=split, name=name, text_column_name="caption")

    def __getitem__(self, index):
        suite = self.get_suite(index)

        if "test" in self.split or 'val' in self.split:
            iid = self.table["image_id"][index].as_py()
            # iid = int(iid.split(".")[0].split("_")[-1])
            suite.update({"iid": iid})

        return suite



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


class IuxrayMultiImageDataset(BaseDataset):
    def __getitem__(self, idx):
        example = self.examples[idx]
        image_id = example['id']
        image_path = example['image_path']
        array = image_id.split('-')
        modified_id = array[0] + '-' + array[1]
        label = np.array(self.labels[modified_id]).astype(np.float32)
        # image_path = example['image_path']
        # image_1 = Image.open(os.path.join(self.image_dir, image_path[0])).convert('RGB')
        # image_2 = Image.open(os.path.join(self.image_dir, image_path[1])).convert('RGB')
        if 'images' in example:
            image_1, image_2 = example['images'][0], example['images'][1]
        else:
            image_1 = Image.open(os.path.join(self.image_dir, image_path[0])).convert('RGB')
            image_2 = Image.open(os.path.join(self.image_dir, image_path[1])).convert('RGB')
            self.examples[idx]['images'] = [image_1, image_2]
        if self.transform is not None:
            image_1 = self.transform(image_1)
            image_2 = self.transform(image_2)
        image = torch.stack((image_1, image_2), 0)
        report_ids = example['ids']
        report_masks = example['mask']
        seq_length = len(report_ids)
        sample = (image_id, image, report_ids, report_masks, seq_length, label)
        return sample


class MimiccxrSingleImageDataset(BaseDataset):
    def __getitem__(self, idx):
        example = self.examples[idx]
        image_id = example['id']
        image_path = example['image_path']
        label = np.array(self.labels[image_id]).astype(np.float32)
        try:
            image = Image.open(os.path.join(self.image_dir, image_path[0])).convert('RGB')
        # image = Image.open(os.path.join(self.image_dir, image_path[0])).convert('RGB')
        # image = cv2.cvtColor(cv2.imread(os.path.join(self.image_dir, image_path[0])), cv2.COLOR_BGR2RGB)
        # image = Image.fromarray(image)
        except IOError:
            print('image path', image_path[0])

        if self.transform is not None:
            image = self.transform(image)
        report_ids = example['ids']
        report_masks = example['mask']
        seq_length = len(report_ids)
        sample = (image_id, image, report_ids, report_masks, seq_length, label)
        return sample


class ChexPert(Dataset):
    def __init__(self, label_path, cfg, mode='train', transform=None):
        self.cfg = cfg
        self._label_header = None
        self._image_paths = []
        self._labels = []
        self._mode = mode
        self.dict = [{'1.0': '1', '': '0', '0.0': '0', '-1.0': '0'},
                     {'1.0': '1', '': '0', '0.0': '0', '-1.0': '1'}, ]
        self.transform = transform
        with open(label_path) as f:
            header = f.readline().strip('\n').split(',')
            self._label_header = [
                header[7],
                header[10],
                header[11],
                header[13],
                header[15]]
            for line in f:
                labels = []
                fields = line.strip('\n').split(',')
                # print('111', fields)
                image_path = fields[0]
                flg_enhance = False
                for index, value in enumerate(fields[5:]):
                    if index == 5 or index == 8:
                        labels.append(self.dict[1].get(value))
                        if self.dict[1].get(
                                value) == '1' and \
                                self.cfg.enhance_index.count(index) > 0:
                            flg_enhance = True
                    elif index == 2 or index == 6 or index == 10:
                        labels.append(self.dict[0].get(value))
                        if self.dict[0].get(
                                value) == '1' and \
                                self.cfg.enhance_index.count(index) > 0:
                            flg_enhance = True
                    else:
                        labels.append(self.dict[1].get(value))
                        if self.dict[1].get(
                                value) == '1' and \
                                self.cfg.enhance_index.count(index) > 0:
                            flg_enhance = True
                # labels = ([self.dict.get(n, n) for n in fields[5:]])
                labels = list(map(int, labels))
                image_path = os.path.join('./data', image_path)
                self._image_paths.append(image_path)
                assert os.path.exists(image_path), image_path
                self._labels.append(labels)
                # if flg_enhance and self._mode == 'train':
                #     for i in range(self.cfg.enhance_times):
                #         self._image_paths.append(image_path)
                #         self._labels.append(labels)
        self._num_image = len(self._image_paths)

    def __len__(self):
        return self._num_image

    def __getitem__(self, idx):
        image = Image.open(self._image_paths[idx]).convert('RGB')
        # image = Image.fromarray(image)
        # if self._mode == 'train':
        #    image = GetTransforms(image, type=self.cfg.use_transforms_type)
        # image = np.array(image)
        # image = transform(image, self.cfg)
        if self.transform:
            image = self.transform(image)

        labels = np.array(self._labels[idx]).astype(np.float32)

        path = self._image_paths[idx]

        if self._mode == 'train' or self._mode == 'dev':
            return (image, labels)
        elif self._mode == 'test':
            return (image, path)
        elif self._mode == 'heatmap':
            return (image, path, labels)
        else:
            raise Exception('Unknown mode : {}'.format(self._mode))


class IuxrayMultiImageClsDataset(Dataset):
    def __init__(self, args, split, transform=None, vis=False):
        self.image_dir = os.path.join(args.data_dir, args.dataset_name, 'images')
        self.ann_path = os.path.join(args.data_dir, args.dataset_name, 'annotation.json')
        self.split = split
        self.ann = json.loads(open(self.ann_path, 'r').read())
        self.examples = self.ann[self.split]
        self.labels_path = os.path.join(args.data_dir, args.dataset_name, args.label_path)
        self.labels = json.loads(open(self.labels_path, 'r').read())
        self.transform = transform
        self._labels = []
        for e in self.examples:
            img_id = e['id']
            array = img_id.split('-')
            modified_id = array[0] + '-' + array[1]
            self._labels.append(self.labels[modified_id])
        self.vis = vis
        # self._labels = [self.labels[e['id']] for e in self.examples]

    def __getitem__(self, idx):
        example = self.examples[idx]
        image_id = example['id']
        array = image_id.split('-')
        modified_id = array[0] + '-' + array[1]
        label = np.array(self.labels[modified_id]).astype(np.float32)
        image_path = example['image_path']
        image_1 = Image.open(os.path.join(self.image_dir, image_path[0])).convert('RGB')
        image_2 = Image.open(os.path.join(self.image_dir, image_path[1])).convert('RGB')
        if self.transform is not None:
            image_1 = self.transform(image_1)
            image_2 = self.transform(image_2)
        image = torch.stack((image_1, image_2), 0)
        # sample = (image_id, image, report_ids, report_masks, seq_length)
        # sample = (image, label)
        if self.vis:
            return image, image_path, label
        else:
            return image, label

    def __len__(self):
        return len(self.examples)


class MimiccxrSingleImageClsDataset(BaseDataset):
    def __init__(self, args, split, transform=None, vis=False):
        self.image_dir = os.path.join(args.data_dir, args.dataset_name, 'images')
        self.ann_path = os.path.join(args.data_dir, args.dataset_name, 'annotation.json')
        self.split = split
        self.ann = json.loads(open(self.ann_path, 'r').read())
        self.examples = self.ann[self.split]
        self.labels_path = os.path.join(args.data_dir, args.dataset_name, args.label_path)
        self.labels = json.loads(open(self.labels_path, 'r').read())
        self.transform = transform
        self._labels = [self.labels[e['id']] for e in self.examples]
        self.vis = vis

    def __getitem__(self, idx):
        example = self.examples[idx]
        image_id = example['id']
        label = np.array(self.labels[image_id]).astype(np.float32)
        image_path = example['image_path']
        image = Image.open(os.path.join(self.image_dir, image_path[0])).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        # report_ids = example['ids']
        # report_masks = example['mask']
        # seq_length = len(report_ids)
        # sample = (image_id, image, report_ids, report_masks, seq_length)
        if self.vis:
            return image, image_path, label
        else:
            return image, label

    def __len__(self):
        return len(self.examples)

#
# class CheXpertDataSet(Dataset):
#     def __init__(self, image_list_file, transform=None, policy="ones"):
#         """
#         image_list_file: path to the file containing images with corresponding labels.
#         transform: optional transform to be applied on a sample.
#         Upolicy: name the policy with regard to the uncertain labels
#         """
#         image_names = []
#         labels = []
#         self.dict = [{'1.0': '1', '': '0', '0.0': '0', '-1.0': '0'},
#                      {'1.0': '1', '': '0', '0.0': '0', '-1.0': '1'}, ]
#
#         with open(image_list_file, "r") as f:
#             csvReader = csv.reader(f)
#             next(csvReader, None)
#             k = 0
#             for line in csvReader:
#                 k += 1
#                 image_name = line[0]
#                 label = line[5:]
#
#                 for i in range(14):
#                     if label[i]:
#                         a = float(label[i])
#                         if a == 1:
#                             label[i] = 1
#                         elif a == -1:
#                             if policy == "ones":
#                                 label[i] = 1
#                             elif policy == "zeroes":
#                                 label[i] = 0
#                             else:
#                                 label[i] = 0
#                         else:
#                             label[i] = 0
#                     else:
#                         label[i] = 0
#
#                 image_names.append('../' + image_name)
#                 labels.append(label)
#
#         self.image_names = image_names
#         self.labels = labels
#         self.transform = transform
#
#     def __getitem__(self, index):
#         """Take the index of item and returns the image and its labels"""
#
#         image_name = self.image_names[index]
#         image = Image.open(image_name).convert('RGB')
#         label = self.labels[index]
#         if self.transform is not None:
#             image = self.transform(image)
#         return image, torch.FloatTensor(label)
#
#     def __len__(self):
#         return len(self.image_names)

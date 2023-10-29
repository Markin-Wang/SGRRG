import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
from data import *
from torch.utils.data import DistributedSampler
import torch.distributed as dist
from modules.balanced_sampler import MultilabelBalancedRandomSampler
import random
from collections import defaultdict
# from mmcv.transforms import LoadImageFromFile, Resize,CenterCrop,Normalize,ImageToTensor,RandomCrop
import torchvision
from config import cgnome_id2cat, cgnome_cumcat

torchvision.disable_beta_transforms_warning()
import torchvision.transforms.v2 as transforms


class R2DataLoader(DataLoader):
    def __init__(self, config, tokenizer, split, shuffle, vis=False):
        self.config = config
        self.dataset_name = config['dataset_name']
        self.batch_size = config['batch_size'] if split == 'train' else config['eval_batch_size']
        self.shuffle = shuffle
        self.num_workers = config['num_workers']
        self.split = split
        self.drop_last = True if split == 'train' else False
        self.vis = vis
        self.test = config['test']
        self.att_cls = config['att_cls']
        g = torch.Generator()
        g.manual_seed(config['seed'])

        if split == 'train':
            self.transform = {'common_aug': transforms.Compose([
                transforms.Resize((config['image_size'], config['image_size'])),
                transforms.RandomRotation(config['rotate_degree']),
                # transforms.RandomCrop(224),
                # transforms.ToImageTensor(),  # note this does not scale the image
                # transforms.ConvertImageDtype(torch.float32),
                # transforms.Normalize((123.675, 116.28, 103.53),
                #                      (58.395, 57.12, 57.375))
            ]),
                'norm_to_tensor': transforms.Compose([
                    transforms.ToImageTensor(),  # note this does not scale the image
                    transforms.ConvertImageDtype(torch.float32),  # this operation scale the value from 255 to [0,1]
                    transforms.Normalize((0.485, 0.456, 0.406),
                                         (0.229, 0.224, 0.225))
                ]),
            }
        else:
            self.transform = {'common_aug': transforms.Compose([
                transforms.Resize((config['image_size'], config['image_size'])),
                # transforms.RandomCrop(224),
                # transforms.ToImageTensor(),  # note this does not scale the image
                # transforms.ConvertImageDtype(torch.float32),
                # transforms.Normalize((123.675, 116.28, 103.53),
                #                      (58.395, 57.12, 57.375))
            ]),
                'norm_to_tensor': transforms.Compose([
                    transforms.ToImageTensor(),  # note this does not scale the image
                    transforms.ConvertImageDtype(torch.float32),  # this operation scale the value from 255 to [0,1]
                    transforms.Normalize((0.485, 0.456, 0.406),
                                         (0.229, 0.224, 0.225))
                ]),
            }

            # self.transform = [
            #     Resize(scale=(config['image_size'], config['image_size'])),
            #     Normalize(),
            #     ImageToTensor(),
            # ]

        if self.dataset_name == 'iu_xray':
            self.dataset = IuxrayMultiImageDatasetArrow(config=self.config, tokenizer=tokenizer, split=self.split,
                                                        transform=self.transform)
        elif self.dataset_name.startswith('mimic'):
            self.dataset = MIMICMultiImageDatasetArrow(config=self.config, tokenizer=tokenizer, split=self.split,
                                                       transform=self.transform)
        elif self.dataset_name.startswith('cxr_gnome'):
            self.dataset = CXRGenomeDatasetArrow(config=self.config, tokenizer=tokenizer, split=self.split,
                                                 transform=self.transform)
        else:
            raise NotImplementedError

        num_tasks = dist.get_world_size()
        global_rank = dist.get_rank()
        self.sampler = torch.utils.data.DistributedSampler(
            self.dataset, num_replicas=num_tasks, rank=global_rank, shuffle=self.shuffle
        )

        if config['balanced']:
            if split == 'train' and not self.vis:
                print('Balanced sampler is established for ' + split + ' dataset.')
                self.sampler = MultilabelBalancedRandomSampler(np.array(self.dataset._labels))
                self.init_kwargs = {
                    'dataset': self.dataset,
                    'batch_size': self.batch_size,
                    'sampler': self.sampler,
                    'num_workers': self.num_workers,
                    'pin_memory': True,
                    'drop_last': self.drop_last,
                    'collate_fn': self.collate_fn,
                    'worker_init_fn': seed_worker,
                    #'prefetch_factor': self.batch_size // self.num_workers * 2
                }
            else:
                self.init_kwargs = {
                    'dataset': self.dataset,
                    # 'sampler': self.sampler,
                    'batch_size': self.batch_size,
                    'shuffle': shuffle,
                    'num_workers': self.num_workers,
                    'pin_memory': True,
                    'drop_last': self.drop_last,
                    'collate_fn': self.collate_fn,
                    'worker_init_fn': seed_worker,
                    'generator': g,
                    #'prefetch_factor': self.batch_size // self.num_workers * 2
                }

        else:
            self.init_kwargs = {
                'dataset': self.dataset,
                'sampler': self.sampler,
                'batch_size': self.batch_size,
                # 'shuffle':shuffle,
                'collate_fn': self.collate_fn,
                'worker_init_fn': seed_worker,
                'num_workers': self.num_workers,
                'pin_memory': True,
                'drop_last': self.drop_last,
                #'prefetch_factor': self.batch_size // self.num_workers * 2
            }

        # num_tasks = dist.get_world_size()
        # global_rank = dist.get_rank()
        #
        # self.sampler = DistributedSampler(self.dataset, num_replicas=num_tasks,
        #                                   rank=global_rank, shuffle=self.shuffle)

        super().__init__(**self.init_kwargs)

    @staticmethod
    def collate_fn(batch):
        keys = ['img_id', 'image', 'text', 'mask', 'seq_length']  # data used

        if 'boxes' in batch[0].keys():
            keys.extend(['box_labels', 'boxes'])

        if 'region_labels' in batch[0].keys():
            keys.extend(['region_labels'])

        if 'box_masks' in batch[0].keys():
            keys.extend(['box_masks'])

        if 'attribute_labels' in batch[0].keys():
            keys.extend(['attribute_labels'])

        if 'attribute_label_dicts' in batch[0].keys():
            keys.extend(['attribute_label_dicts'])

        batch_dict = {key: [sample[key] for sample in batch] for key in keys}

        reports_ids, reports_masks, seq_lengths, = batch_dict['text'], batch_dict['mask'], batch_dict['seq_length']

        batch_dict['image'] = torch.stack(batch_dict['image'], 0)
        max_seq_length = max(seq_lengths)

        targets = np.zeros((len(reports_ids), max_seq_length), dtype=int)
        targets_masks = np.zeros((len(reports_ids), max_seq_length), dtype=int)

        for i, report_ids in enumerate(reports_ids):
            targets[i, :len(report_ids)] = report_ids

        for i, report_masks in enumerate(reports_masks):
            targets_masks[i, :len(report_masks)] = report_masks

        batch_dict['text'] = torch.LongTensor(targets)
        batch_dict['mask'] = torch.FloatTensor(targets_masks)

        if 'boxes' in batch[0].keys():
            boxes, labels = batch_dict['boxes'], batch_dict['box_labels']
            boxes_, labels_ = [], []
            for i, (box, label) in enumerate(zip(boxes, labels)):
                num_box = len(box)
                box_with_id = torch.zeros(num_box, 5)
                box_with_id[:, 0] = i
                box_with_id[:, 1:] = box
                boxes_.append(box_with_id)
                labels_.append(torch.from_numpy(label))

            batch_dict['boxes'] = torch.cat(boxes_, dim=0)
            batch_dict['box_labels'] = torch.cat(labels_, dim=0)

        if 'region_labels' in batch[0].keys():
            batch_dict['region_labels'] = torch.cat(batch_dict['region_labels'], dim=0)

        if 'box_masks' in batch[0].keys():
            batch_dict['box_masks'] = torch.cat(batch_dict['box_masks'], dim=0)

        if 'attribute_labels' in batch[0].keys():
            attribute_labels = []
            selected_box_labels = batch_dict['box_labels'][batch_dict['box_masks']]
            selected_boxes_bsid = batch_dict['boxes'][batch_dict['box_masks'], 0]
            attribute_masks = []
            attribute_ids = []
            max_att = -1
            for i, attribute_label in enumerate(batch_dict['attribute_labels']):
                i_box_labels = selected_box_labels[selected_boxes_bsid == i]
                if len(attribute_label) == 0:
                    attribute_masks.append(torch.ones(i_box_labels.size(0)))
                    continue
                else:
                    attribute_masks.append(torch.zeros(i_box_labels.size(0)))
                for box_label in i_box_labels:
                    temp_label = attribute_label[box_label.item()]
                    max_att = max(max_att, len(temp_label))
                    # attribute_ids.append(temp_label)
                    attribute_ids.append(np.array(temp_label) + cgnome_cumcat[box_label.item()])
                    # attribute_label_ = torch.zeros(1,849) # 849 attributes
                    attribute_label_ = torch.zeros(1, cgnome_id2cat[box_label.item()])
                    attribute_label_[0, temp_label] = 1.0
                    attribute_labels.append(attribute_label_)
            if attribute_labels:
                # batch_dict['attribute_labels'] = torch.cat(attribute_labels,dim=0)
                batch_dict['attribute_labels'] = torch.cat(attribute_labels, dim=-1)
                # print(1111,batch_dict['attribute_labels'].shape, (batch_dict['attribute_labels']==1).sum()/batch_dict['attribute_labels'].shape[1])
                # around 8% positive labels
                attribute_ids_ = torch.full((len(attribute_labels), max_att), -10000, dtype=torch.long)  # att_pad_idx is -1e4
                for i, att_id in enumerate(attribute_ids):
                    attribute_ids_[i, :len(att_id)] = torch.LongTensor(att_id)
                batch_dict['attribute_ids'] = attribute_ids_
            else:
                batch_dict['attribute_labels'] = []
            batch_dict['attribute_masks'] = torch.cat(attribute_masks, dim=0)

        if 'attribute_label_dicts' in batch[0].keys():
            attribute_labels = []
            selected_box_labels = batch_dict['box_labels'][batch_dict['box_masks']]
            selected_boxes_bsid = batch_dict['boxes'][batch_dict['box_masks'], 0]
            for i, attribute_label in enumerate(batch_dict['attribute_label_dicts']):
                i_box_labels = selected_box_labels[selected_boxes_bsid == i]
                cur_attribute_labels = {}
                for box_label in i_box_labels:
                    temp_label = attribute_label[box_label.item()]
                    # attribute_ids.append(temp_label)
                    # attribute_label_ = torch.zeros(1,849) # 849 attributes
                    attribute_label_ = torch.zeros(1, cgnome_id2cat[box_label.item()])
                    attribute_label_[0, temp_label] = 1.0
                    cur_attribute_labels[box_label.item()] = attribute_label_
                attribute_labels.append(cur_attribute_labels)
                # batch_dict['attribute_labels'] = torch.cat(attribute_labels,dim=0)
            batch_dict['attribute_label_dicts'] = attribute_labels

        return batch_dict


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

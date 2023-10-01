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

torchvision.disable_beta_transforms_warning()
import torchvision.transforms.v2 as transforms


class R2DataLoader(DataLoader):
    def __init__(self, config, tokenizer, split, shuffle, vis=False):
        self.config = config
        self.cls = config['cls']
        self.dataset_name = config['dataset_name']
        self.batch_size = config['batch_size']
        self.shuffle = shuffle
        self.num_workers = config['num_workers']
        self.split = split
        self.drop_last = True if split == 'train' else False
        self.vis = vis
        self.test = config['test']
        self.att_cls = config['att_cls']
        if split != 'train' and self.batch_size > 200: self.batch_size //= 2
        g = torch.Generator()
        g.manual_seed(config['seed'])

        if split == 'train':
            if config['randaug']:
                print('Random applied transformation is utilized for ' + split + ' dataset.')
                self.transform = transforms.Compose([
                    transforms.Resize(256),
                    transforms.RandomCrop(224),
                    transforms.RandomApply([
                        transforms.RandomRotation(10, interpolation=transforms.InterpolationMode.BICUBIC),
                        # transforms.RandomAffine(0, shear=10, interpolation=transforms.InterpolationMode.BICUBIC),
                        transforms.RandomAffine(0, scale=(0.8, 1.2),
                                                interpolation=transforms.InterpolationMode.BICUBIC)
                    ]),
                    # transforms.RandomHorizontalFlip(),
                    # transforms.RandomPerspective(distortion_scale=0.2),
                    # transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0),
                    transforms.ToTensor(),
                    # transforms.RandomErasing(scale=(0.02, 0.16), ratio=(0.3, 1.6)),
                    transforms.Normalize((0.485, 0.456, 0.406),
                                         (0.229, 0.224, 0.225))])
            else:
                # if self.att_cls:
                #     self.transform = [
                #         Resize(scale=(config['image_size'], config['image_size'])),
                #         Normalize(),
                #         ImageToTensor(),
                #     ]
                # else:
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
                        transforms.ConvertImageDtype(torch.float32), # this operation scale the value from 255 to [0,1]
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
                    transforms.ConvertImageDtype(torch.float32), # this operation scale the value from 255 to [0,1]
                    transforms.Normalize((0.485, 0.456, 0.406),
                                         (0.229, 0.224, 0.225))
                ]),
            }

            # self.transform = [
            #     Resize(scale=(config['image_size'], config['image_size'])),
            #     Normalize(),
            #     ImageToTensor(),
            # ]

        if self.dataset_name == 'iu_xray' and not self.cls:
            self.dataset = IuxrayMultiImageDatasetArrow(config=self.config, tokenizer=tokenizer, split=self.split,
                                                        transform=self.transform)
        elif self.dataset_name == 'iu_xray' and self.cls:
            self.dataset = IuxrayMultiImageClsDataset(self.config, tokenizer, self.split, transform=self.transform,
                                                      vis=self.vis)
        elif self.dataset_name.startswith('mimic') and not self.cls:
            self.dataset = MIMICMultiImageDatasetArrow(config=self.config, tokenizer=tokenizer, split=self.split,
                                                       transform=self.transform)
        elif self.dataset_name.startswith('mimic') and self.cls:
            self.dataset = MimiccxrSingleImageClsDataset(self.config, self.split, transform=self.transform,
                                                         vis=self.vis)
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
                    'prefetch_factor': self.batch_size // self.num_workers * 2
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
                    'prefetch_factor': self.batch_size // self.num_workers * 2
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
                'prefetch_factor': self.batch_size // self.num_workers * 2
            }

        # num_tasks = dist.get_world_size()
        # global_rank = dist.get_rank()
        #
        # self.sampler = DistributedSampler(self.dataset, num_replicas=num_tasks,
        #                                   rank=global_rank, shuffle=self.shuffle)

        super().__init__(**self.init_kwargs)

    @staticmethod
    def collate_fn(batch):
        keys = ['img_id', 'image', 'img_labels', 'text', 'mask', 'seq_length']  # data used
        batch_dict = {key: [sample[key] for sample in batch] for key in keys}

        reports_ids, reports_masks, seq_lengths, labels = batch_dict['text'], batch_dict['mask'], \
                                                          batch_dict['seq_length'], batch_dict['img_labels']
        batch_dict['image'] = torch.stack(batch_dict['image'], 0)
        max_seq_length = max(seq_lengths)
        labels = np.array(labels)

        targets = np.zeros((len(reports_ids), max_seq_length), dtype=int)
        targets_masks = np.zeros((len(reports_ids), max_seq_length), dtype=int)

        for i, report_ids in enumerate(reports_ids):
            targets[i, :len(report_ids)] = report_ids

        for i, report_masks in enumerate(reports_masks):
            targets_masks[i, :len(report_masks)] = report_masks

        batch_dict['text'] = torch.LongTensor(targets)
        batch_dict['mask'] = torch.FloatTensor(targets_masks)
        batch_dict['img_labels'] = torch.FloatTensor(labels)

        return batch_dict


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

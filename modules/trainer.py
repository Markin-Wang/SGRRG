import os
from abc import abstractmethod

import time
import torch
import datetime
import pandas as pd
from numpy import inf
from tqdm import tqdm
from modules.utils import auto_resume_helper, get_grad_norm, reduce_tensor, get_loss_and_list
from modules.weighted_mesloss import Weighted_MSELoss
from torch.cuda.amp import GradScaler, autocast
from timm.utils import AverageMeter
import torch.distributed as dist
from .utils import clip_grad_norm_, calculate_auc, get_region_mask, gather_preds_and_gts
import numpy as np
from modules.beam_search import BeamSearch
from modules.loss import AsymmetricLoss, AsymmetricLossOptimized
from collections import defaultdict
from config import cgnome_id2cat, categories
import json
import glob


# from modules.utils import get_grad_norm
# import torch.distributed as dist

def nan_hook(self, inp, output):
    if not isinstance(output, tuple):
        outputs = [output]
    else:
        outputs = output

    for i, out in enumerate(outputs):
        if out is None or not isinstance(out, torch.Tensor): continue
        nan_mask = torch.isnan(out)
        if nan_mask.any():
            print("In", self.__class__.__name__)
            raise RuntimeError(f"Found NAN in output {i} at indices: ", nan_mask.nonzero(), "where:",
                               out[nan_mask.nonzero()[:, 0].unique(sorted=True)])


class BaseTrainer(object):
    def __init__(self, model, criterion, metric_ftns, optimizer, lr_scheduler, logger, config):
        self.config = config

        # setup GPU device if available, move model into configured device
        self.device, device_ids = self._prepare_device(config['n_gpu'])
        self.lr_scheduler = lr_scheduler
        self.clip_value = config['clip_value']
        self.use_amp = config['use_amp']
        self.test_after = config['test_after']
        self.early_stop = config['early_stop']
        self.local_rank = config['local_rank']
        self.att_pad_idx = config['att_pad_idx']
        self.pad_idx = config['pad_idx']
        # if args.debug:
        #     for submodule in model.modules():
        #         submodule.register_forward_hook(nan_hook)
        self.model = model

        self.start_eval = config['start_eval']
        # for name, module in self.model.named_modules():
        #     module.register_backward_hook(get_activations(name))
        self.optimizer = optimizer
        self.scaler = GradScaler(enabled=self.use_amp, init_scale=256)
        self.clip_option = config['clip_option']
        self.max_seq_length = config['max_seq_length']

        self.att_cls = config['att_cls']  # attribute classification
        self.region_cls = config['region_cls']
        self.dis_cls = config['dis_cls']
        self.disr_opt = config['disr_opt']

        self.use_focal_ls_r = config['use_focal_ls_r']
        self.use_focal_ls_a = config['use_focal_ls_a']
        self.use_focal_ls_d = config['use_focal_ls_d']
        self.use_focal_ls_dr = config['use_focal_ls_dr']
        self.orthogonal_ls = config['orthogonal_ls']
        self.orthogonal_ls_w = config['orthogonal_ls_w']
        self.clip = config['clip']
        self.clip_w = config['clip_w']

        if self.region_cls:
            if self.use_focal_ls_r:
                self.region_cls_criterion = AsymmetricLossOptimized(gamma_neg=2, gamma_pos=2, clip=0.0,
                                                                    reduction='mean')
            else:
                self.region_cls_criterion = torch.nn.BCEWithLogitsLoss()
            self.region_cls_w = config['region_cls_w']

        if self.att_cls:
            if self.use_focal_ls_a:
                self.att_cls_criterion = AsymmetricLossOptimized(gamma_neg=2, gamma_pos=2, clip=0.0, reduction='mean')
            else:
                self.att_cls_criterion = torch.nn.BCEWithLogitsLoss()
            self.att_cls_w = config['att_cls_w']

            if self.disr_opt == 'cls':
                if self.use_focal_ls_dr:
                    self.disr_cls_criterion = AsymmetricLossOptimized(gamma_neg=2, gamma_pos=2, clip=0.0,
                                                                      reduction='mean')
                else:
                    self.disr_cls_criterion = torch.nn.BCEWithLogitsLoss()

            self.disr_w = config['disr_w']

        if self.dis_cls:
            if self.use_focal_ls_d:
                self.dis_cls_criterion = AsymmetricLossOptimized(gamma_neg=2, gamma_pos=2, clip=0.0, reduction='mean')
            else:
                self.dis_cls_criterion = torch.nn.BCEWithLogitsLoss()
            self.dis_cls_w = config['dis_cls_w']

        self.use_sg = config['use_sg']
        self.criterion = criterion
        self.metric_ftns = metric_ftns

        self.logger = logger
        self.use_new_bs = config['use_new_bs']

        self.epochs = config['epochs']
        self.save_period = config['save_period']

        self.mnt_mode = 'max'
        self.mnt_metric = 'val_' + config['monitor_metric']
        self.mnt_metric_test = 'test_' + config['monitor_metric']
        assert self.mnt_mode in ['min', 'max']

        self.mnt_val_best = inf if self.mnt_mode == 'min' else -inf
        self.mnt_test_best = inf if self.mnt_mode == 'min' else -inf
        self.early_exit = config['early_exit']

        self.start_epoch = 1
        self.checkpoint_dir = os.path.join(config['output'], config['dataset_name'], config['exp_name'])
        self.best_epoch = 0
        self.print_freq = config['print_freq']

        # keys to cuda
        self.keys = set(
            ['image', 'text', 'mask', 'boxes', 'box_labels', 'box_abnormal_labels', 'box_masks', 'region_labels',
             'attribute_labels', 'attribute_ids', 'disease_labels'])

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir, exist_ok=True)

        if config['resume']:
            self._resume_checkpoint(self.checkpoint_dir)

        self.best_recorder = {'val': {self.mnt_metric: self.mnt_val_best},
                              'test': {self.mnt_metric_test: self.mnt_test_best}}

        self.beam_search = BeamSearch(config)

    @abstractmethod
    def _train_epoch(self, epoch):
        raise NotImplementedError

    @abstractmethod
    def _valid(self, epoch, split):
        raise NotImplementedError

    @abstractmethod
    def test(self):
        raise NotImplementedError

    def train(self):

        not_improved_count = 0
        for epoch in range(self.start_epoch, self.epochs + 1):
            start = time.time()
            log = self._train_epoch(epoch)

            # evaluate model performance according to configured metric,
            if epoch > self.start_eval:
                log.update(self._valid(epoch, 'val')[0])
                if not self.test_after:
                    log.update(self._valid(epoch, 'test')[0])
                # save logged informations into log dict
                log = self._synchronize_data(log)
                # synchronize log in different gpu
                # self.logger.info('Evaluation completed.')
                # log = self._broadcast_data(log)

                log['epoch'] = epoch

                # print logged informations to the screen
                for key, value in log.items():
                    self.logger.info('\t{:15s}: {}'.format(str(key), value))

                improved = self._record_best(log)

                best = False

                if improved:
                    not_improved_count = 0
                    best = True
                    self.best_epoch = epoch
                else:
                    not_improved_count += 1

                self.logger.info('current best model in: {}'.format(self.best_epoch))

                if dist.get_rank() == self.local_rank and epoch % self.save_period == 0:
                    # save best checkpoint as model_best
                    self._save_checkpoint(epoch, save_best=best)
                    self._write_log_to_file(log, epoch)

            torch.cuda.synchronize()
            epoch_time = time.time() - start
            self.logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")

            if not_improved_count > self.early_stop:
                self.logger.info("Validation performance didn\'t improve for {} epochs. " "Training stops.".format(
                    self.early_stop))
                break

            if self.use_sg:
                zero_count = self.model.module.scene_graph_encoder.zero_count
                self.model.module.scene_graph_encoder.zero_count = 0
                self.logger.info(f'There are {zero_count} samples without any region selected in this epoch.')
                # add 1 avoid divide zero
                self.logger.info(
                    f'{self.model.module.scene_graph_encoder.zero_att_count / (self.model.module.scene_graph_encoder.all_box_count + 1) * 100:.2f}% boxes without any attributes predicted.')
                self.model.module.scene_graph_encoder.zero_att_count = 0
                self.model.module.scene_graph_encoder.all_box_count = 0
        if dist.get_rank() == self.local_rank:
            self._print_best()
            self._print_best_to_file()

    def _synchronize_data(self, log):
        pairs = [[k, v] for k, v in log.items()]
        keys = [x[0] for x in pairs]
        values = torch.Tensor([x[1] for x in pairs]).to(self.model.device)
        values = reduce_tensor(values)
        log.update({k: v.item() for k, v in zip(keys, values)})
        return log

    def _broadcast_data(self, log):
        # pairs = [[k, v] for k, v in log.items()]
        # keys = [x[0] for x in pairs]
        if dist.get_rank() == self.local_rank:
            object = [log]
        else:
            object = [None]
        torch.distributed.broadcast_object_list(object, src=self.local_rank)
        return object[0]

    def _print_best_to_file(self):
        crt_time = time.asctime(time.localtime(time.time()))
        self.best_recorder['val']['time'] = crt_time
        self.best_recorder['test']['time'] = crt_time
        self.best_recorder['val']['seed'] = self.config['seed']
        self.best_recorder['test']['seed'] = self.config['seed']
        self.best_recorder['val']['best_model_from'] = 'val'
        self.best_recorder['test']['best_model_from'] = 'test'
        record_dir = os.path.join(self.config['record_dir'], self.config['dataset_name'])
        if not os.path.exists(record_dir):
            os.makedirs(record_dir)
        record_path = os.path.join(record_dir, self.config['exp_name'] + '.csv')
        if not os.path.exists(record_path):
            record_table = pd.DataFrame()
        else:
            record_table = pd.read_csv(record_path)
        record_table = pd.concat([record_table, pd.DataFrame([self.best_recorder['val']]),
                                  pd.DataFrame([self.best_recorder['test']])], ignore_index=True)
        # record_table = record_table.concat(self.best_recorder['val'], ignore_index=True)
        # record_table = record_table.concat(self.best_recorder['test'], ignore_index=True)
        record_table.to_csv(record_path, index=False)

    def _write_log_to_file(self, log, epoch):
        for key, value in log.items():
            self.writer.add_scalar(f'data/{key}', value, epoch)

    def _prepare_device(self, n_gpu_use):
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            print("Warning: There\'s no GPU available on this machine," "training will be performed on CPU.")
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            print(
                "Warning: The number of GPU\'s configured to use is {}, but only {} are available " "on this machine.".format(
                    n_gpu_use, n_gpu))
            n_gpu_use = n_gpu
        device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
        list_ids = list(range(n_gpu_use))
        return device, list_ids

    def _save_checkpoint(self, epoch, save_best=False):
        if not save_best: return
        state = {
            'epoch': epoch,
            'state_dict': self.model.module.state_dict(),
            'lr_scheduler': self.lr_scheduler.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scaler': self.scaler.state_dict(),
            # 'amp':amp.state_dict() if self.amp_opt_level!='O0' else None,
            'monitor_best': self.mnt_val_best,
            'monitor_test_best': self.mnt_test_best,
            'best_epoch': self.best_epoch,
        }
        filename = os.path.join(self.checkpoint_dir, 'model_best.pth')
        # filename = os.path.join(self.checkpoint_dir, 'checkpoint'+str(epoch)+'.pth')
        torch.save(state, filename)
        print("Saving checkpoint: {} ...".format(filename))
        # if save_best:
        #     best_path = os.path.join(self.checkpoint_dir, 'model_best.pth')
        #     torch.save(state, best_path)
        #     print("Saving current best: model_best.pth ...")

    def _resume_checkpoint(self, resume_path):
        resume_file = auto_resume_helper(resume_path)
        # resume_path = str(resume_path)
        print("Loading checkpoint: {} ...".format(resume_file))
        checkpoint = torch.load(resume_file)
        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_val_best = checkpoint['monitor_best']
        self.mnt_test_best = checkpoint['monitor_test_best']
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.best_epoch = checkpoint['best_epoch']
        self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        if self.use_amp:
            self.scaler.load_state_dict(checkpoint['scaler'])
        # if self.amp_opt_level!='O0':
        #     amp.load_state_dict(checkpoint['amp'])

        print("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))

    def _record_best(self, log):
        cur_metric = log['val_BLEU_4'] + 0.5 * log['val_METEOR']
        improved_val = (self.mnt_mode == 'min' and cur_metric <= self.mnt_val_best) or \
                       (self.mnt_mode == 'max' and cur_metric > self.mnt_val_best)
        if improved_val:
            self.mnt_val_best = cur_metric
            self.best_recorder['val'].update(log)

        if not self.test_after:
            cur_metric = log['test_BLEU_4'] + 0.5 * log['test_METEOR']

            improved_test = (self.mnt_mode == 'min' and cur_metric <= self.mnt_test_best) or \
                            (self.mnt_mode == 'max' and cur_metric > self.mnt_test_best)
            if improved_test:
                self.best_recorder['test'].update(log)
                self.mnt_test_best = cur_metric
        return improved_val

    def _print_best(self):
        print('exp_name:', self.config['exp_name'])
        print('Best results (w.r.t {}) in validation set:'.format(self.mnt_metric))
        for key, value in self.best_recorder['val'].items():
            print('\t{:15s}: {}'.format(str(key), value))

        print('Best results (w.r.t {}) in test set:'.format(self.mnt_metric))
        for key, value in self.best_recorder['test'].items():
            print('\t{:15s}: {}'.format(str(key), value))


class Trainer(BaseTrainer):
    def __init__(self, model, criterion, metric_ftns, optimizer, lr_scheduler, dm, writer, logger, config):
        super(Trainer, self).__init__(model, criterion, metric_ftns, optimizer, lr_scheduler, logger, config)
        self.dm = dm
        self.train_dataloader = dm['train_dataloader']
        self.val_dataloader = dm['val_dataloader']
        self.test_dataloader = dm['test_dataloader']
        self.writer = writer
        self.config = config
        self.tokenizer = dm['tokenizer']

    def _train_epoch(self, epoch):
        ce_losses = AverageMeter()
        clip_losses = AverageMeter()
        dis_cls_losses = AverageMeter()
        disr_losses = AverageMeter()
        region_cls_losses = AverageMeter()
        attribute_cls_losses = AverageMeter()
        orthogonal_losses = AverageMeter()
        norm_meter = AverageMeter()
        num_steps = len(self.train_dataloader)
        self.model.train()
        self.optimizer.zero_grad()
        device = self.model.device
        img2attinfo = {}

        with tqdm(desc='Epoch %d - train' % epoch, disable=dist.get_rank() != self.local_rank,
                  unit='it', total=len(self.train_dataloader)) as pbar:
            for batch_idx, data in enumerate(self.train_dataloader):
                # attribute_labels, attribute_ids = data['attribute_labels'], data['attribute_ids']
                # img_ids = data['img_id']
                # img2attinfo.update(
                #     {img_ids[i]:{'attribute_labels':attribute_labels[i],'attribute_ids':attribute_ids[i]} for i in range(len(img_ids))}
                # )
                # pbar.update()
                # continue

                batch_dict = {key: data[key].to(device, non_blocking=True) for key in data.keys() if key in self.keys}

                self.optimizer.zero_grad()
                with autocast(dtype=torch.float16):
                    # region logits is None when att_cls disabled
                    output = self.model(batch_dict, split='train')
                    rrg_preds = output['rrg_preds']
                    loss = self.criterion(rrg_preds, batch_dict['text'], batch_dict['mask'])
                    ce_losses.update(loss.item())

                    if self.dis_cls:
                        dis_logits = output['dis_logits']
                        dis_cls_loss = self.dis_cls_criterion(dis_logits, batch_dict['disease_labels'])
                        loss = loss + self.dis_cls_w * dis_cls_loss
                        dis_cls_losses.update(dis_cls_loss.item())

                    if self.region_cls:
                        region_logits = output['region_logits']
                        region_cls_loss = self.region_cls_criterion(region_logits, batch_dict['region_labels'])
                        loss = loss + self.region_cls_w * region_cls_loss
                        region_cls_losses.update(region_cls_loss.item())

                    if self.att_cls:
                        attribute_logits = output['att_logits']
                        attribute_logits = attribute_logits[attribute_logits != self.att_pad_idx].unsqueeze(0)
                        attribute_cls_loss = self.att_cls_criterion(attribute_logits, batch_dict['attribute_labels'])
                        loss = loss + self.att_cls_w * attribute_cls_loss
                        attribute_cls_losses.update(attribute_cls_loss.item())

                        if self.disr_opt == 'cls':
                            disr_logits = output['disr_logits']
                            disr_labels = batch_dict['box_abnormal_labels'][batch_dict['box_masks']].unsqueeze(
                                -1)  # [bs ,1]
                            disr_loss = self.disr_cls_criterion(disr_logits, disr_labels)
                            loss = loss + self.disr_w * disr_loss
                            disr_losses.update(disr_loss.item())

                        elif self.disr_opt and  self.disr_opt.startswith('con'):
                            # apply contrastive loss
                            disr_loss = output['disr_logits']
                            loss = loss + self.disr_w * disr_loss
                            disr_losses.update(disr_loss.item())
                        elif self.disr_opt is None:
                            pass
                        else:
                            raise NotImplementedError

                    if self.clip:
                        clip_loss = output['clip_loss']
                        loss = loss + self.clip_w * clip_loss
                        clip_losses.update(clip_loss.item())


                    if self.orthogonal_ls:
                        orthogonal_loss = output['orthogonal_ls']
                        loss = loss + self.orthogonal_ls_w * orthogonal_loss
                        orthogonal_losses.update(orthogonal_loss.item())


                self.scaler.scale(loss).backward()

                self.scaler.unscale_(self.optimizer)
                if self.clip_option == 'norm':
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_value)
                elif self.clip_option == 'mynorm':
                    grad_norm = clip_grad_norm_(self.model.parameters(), self.clip_value)
                elif self.clip_option == 'value':
                    torch.nn.utils.clip_grad_value_(self.model.parameters(), self.clip_value)
                    grad_norm = get_grad_norm(self.model.parameters())
                else:
                    grad_norm = get_grad_norm(self.model.parameters())

                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.lr_scheduler.step_update((epoch - 1) * num_steps + batch_idx)

                # self.lr_scheduler.step_update((epoch) * num_steps + batch_idx)
                norm_meter.update(grad_norm)
                # memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
                # pbar.set_postfix(ce=f'{ce_losses.val:.3f} ({ce_losses.avg:.3f})\t',
                #                  dis_cls=f'{dis_cls_losses.val:.3f} ({dis_cls_losses.avg:.3f})\t',
                #                  disr_ls=f'{disr_losses.val:.3f} ({disr_losses.avg:.3f})\t',
                #                  rg_cls=f'{region_cls_losses.val:.3f} ({region_cls_losses.avg:.3f})\t',
                #                  att_cls=f'{attribute_cls_losses.val:.3f} ({attribute_cls_losses.avg:.3f})\t',
                #                  mem=f'mem {memory_used:.0f}MB',
                #                  norm=f'{norm_meter.val:.3f} ({norm_meter.avg:.3f})')
                pbar.update()

                if batch_idx % self.print_freq == 0:
                    memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
                    cur_lr = [round(param_group['lr'],7) for param_group in self.optimizer.param_groups]
                    self.logger.info(
                        f'\n'
                        f'lr:{cur_lr}\t'
                        f'ce_cls:{ce_losses.val:.3f}({ce_losses.avg:.3f})\t'
                        f'dis_cls:{dis_cls_losses.val:.3f}({dis_cls_losses.avg:.3f}) '
                        f'disr_ls:{disr_losses.val:.3f}({disr_losses.avg:.3f}) '
                        f'rg_cls:{region_cls_losses.val:.3f}({region_cls_losses.avg:.3f}) '
                        f'og_ls:{orthogonal_losses.val:.3f}({orthogonal_losses.avg:.3f}) '
                        f'att_cls:{attribute_cls_losses.val:.3f}({attribute_cls_losses.avg:.3f}) '
                        f'clip_ls:{clip_losses.val:.3f}({clip_losses.avg:.3f}) '
                        f'mem {memory_used:.0f}MB '
                        f'norm:{norm_meter.val:.3f}({norm_meter.avg:.3f})'
                        f'\n'
                    )

            # torch.save(img2attinfo,'imgid2attinfo_train.pth')
            # exit()

        log = {'ce_loss': ce_losses.avg,
               'region_loss': region_cls_losses.avg,
               'att_loss': attribute_cls_losses.avg, }
        return log

    # self.writer.add_scalar('data/ce_loss', ce_losses.avg, epoch)
    # self.writer.add_scalar('data/region_cls_loss', region_cls_losses.avg, epoch)
    # self.writer.add_scalar('data/attribute_cls_loss', attribute_cls_loss.avg, epoch)
    # self.writer.add_scalar('data/std_fore', std_fores/len(self.train_dataloader), epoch)
    # self.writer.add_scalar('data/std_attn', std_attns/len(self.train_dataloader), epoch)

    def _valid(self, epoch, split='test'):
        val_ce_losses = AverageMeter()
        val_region_cls_losses = AverageMeter()
        val_dis_cls_losses = AverageMeter()
        val_disr_losses = AverageMeter()
        log = {}
        self.model.eval()
        dataloader = self.val_dataloader if split == 'val' else self.test_dataloader
        device = self.model.device
        region_preds, region_targets = [], []
        dis_preds, dis_targets = [], []
        attribute_preds, attribute_targets = defaultdict(list), defaultdict(list)
        img_ids, img_ids_list, selected_regions, att_recrod_preds = [], [], [], []

        with tqdm(desc=f'Epoch %d - {split}' % epoch, unit='it', total=len(dataloader),
                  disable=dist.get_rank() != self.local_rank) as pbar:
            with torch.no_grad():
                val_gts, val_res = [], []
                for batch_idx, data in enumerate(dataloader):
                    # attribute_labels = data['attribute_label_dicts']
                    # img_ids = data['img_id']
                    # img2attinfo.update(
                    #     {img_ids[i]:attribute_labels[i] for i in range(len(img_ids))}
                    # )
                    # pbar.update()
                    # continue
                    batch_dict = {key: data[key].to(device, non_blocking=True) for key in data.keys() if
                                  key in self.keys}
                    if self.region_cls:
                        # boxes, box_labels, region_labels = data['boxes'].to(device, non_blocking=True), \
                        #                                    data['box_labels'].to(device, non_blocking=True), \
                        #                                    data['region_labels'].to(device, non_blocking=True)
                        region_labels = batch_dict['region_labels']
                        region_masks = get_region_mask(region_labels)
                        region_labels = region_labels[region_masks]

                    with autocast(dtype=torch.float16):
                        output = self.model(batch_dict, split=split)

                        if self.dis_cls:
                            dis_logits = output['dis_logits']
                            dis_probs = output['dis_probs']
                            dis_preds, dis_targets, val_dis_cls_loss  = get_loss_and_list(dis_logits,
                                                                 batch_dict['disease_labels'], dis_preds,
                                                                 dis_targets, self.dis_cls_criterion,
                                                                 cur_probs=dis_probs)
                            val_dis_cls_losses.update(val_dis_cls_loss.item())

                        if self.region_cls:
                            region_logits = output['region_logits'][region_masks]
                            region_probs = output['region_probs'][region_masks]

                            if split == 'test':
                                selected_regions.append([(output['region_probs']>=0.2).cpu(),region_masks.cpu()])
                                img_ids_list.append(data['img_id'])

                            if len(region_labels) > 0:
                                region_preds, region_targets, val_region_cls_loss  = get_loss_and_list(region_logits,
                                                                     region_labels, region_preds,
                                                                     region_targets, self.region_cls_criterion,
                                                                     cur_probs=region_probs)

                                val_region_cls_losses.update(val_region_cls_loss.item())


                        if self.att_cls and split == 'test':
                            att_probs_record = output['att_probs_record']
                            attribute_labels = data['attribute_label_dicts']
                            for bs_id in att_probs_record.keys():
                                for box_label in att_probs_record[bs_id]:
                                    if box_label in attribute_labels[bs_id]:
                                        attribute_preds[box_label].append(
                                            att_probs_record[bs_id][box_label][:cgnome_id2cat[box_label]].unsqueeze(0))
                                        attribute_targets[box_label].append(
                                            attribute_labels[bs_id][box_label])
                            att_recrod_preds.append(att_probs_record)

                        output = self.beam_search.caption_test_step(self.model.module, batch_dict=output)
                        # else:
                        #     output = self.beam_search.sample(self.model.module, patch_feats=output['encoded_img_feats'])

                    if split == 'test':
                        val_res.append(output['preds'])
                        val_gts.append(self._pad(batch_dict['text'][:, 1:]))
                    else:
                        reports = self.tokenizer.decode_batch(output['preds'].cpu().numpy())
                        ground_truths = self.tokenizer.decode_batch(batch_dict['text'][:, 1:].cpu().numpy())
                        val_res.extend(reports)
                        val_gts.extend(ground_truths)
                    img_ids.extend(data['img_id'])

                    memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)

                    pbar.set_postfix(mem=f'mem {memory_used:.0f}MB')
                    pbar.update()

                # torch.save(img2attinfo, f'imgid2attinfo_{split}.pth')
                # return log, val_res, val_gts, img_ids

                if self.dis_cls:
                    dis_auc = calculate_auc(preds=dis_preds.numpy(), targets=dis_targets.long().numpy())
                    log.update({f'{split}_dis_auc': dis_auc, f"{split}_dis_ls": val_dis_cls_losses.avg})

                if self.region_cls:
                    # note in multi-gpu training, results should be reported in 1-gpu test
                    # or gather data from different rank
                    region_auc = calculate_auc(preds=region_preds.numpy(), targets=region_targets.long().numpy())
                    log.update({f'{split}_rg_auc': region_auc, f"{split}_rg_ls": val_region_cls_losses.avg})

                if self.att_cls and split == 'test':
                    att_aucs = []
                    for key in attribute_preds.keys():
                        try:
                            att_preds, att_gts = torch.cat(attribute_preds[key], dim=0), np.concatenate(
                                attribute_targets[key], axis=0)
                            column_sum = att_gts.sum(axis=0)
                            selected_column_ids = column_sum != 0
                            # print(f'{selected_column_ids.sum()} out of {selected_column_ids.shape[0]} categories selected.')
                            att_preds, att_gts = att_preds[:, selected_column_ids], att_gts[:, selected_column_ids]
                            att_auc = calculate_auc(preds=att_preds.numpy(),
                                                    targets=att_gts.astype(np.int64))
                            att_aucs.append(att_auc)
                        except  ValueError:
                            self.logger.info(f'Att calculation on category {categories[key]} fails.')
                            continue
                    att_auc = np.mean(att_aucs) if att_aucs else 0
                    log.update({f'{split}_att_auc': att_auc})

                # ensure the data in each rank is the same to perform evaluation
                if split == 'test':
                    # syncronize data from different split to ensure 100% correct calculation in test set
                    # valid set may have slight difference in multi-gpu training due to the average among all gpus
                    val_res, val_gts = torch.cat(val_res, dim=0), torch.cat(val_gts, dim=0)
                    val_res, val_gts = gather_preds_and_gts(val_res, val_gts)
                    val_res, val_gts = torch.cat(val_res, dim=0), torch.cat(val_gts, dim=0)
                    val_res, val_gts = self.tokenizer.decode_batch(val_res.cpu().numpy()), self.tokenizer.decode_batch(
                        val_gts.cpu().numpy())

                val_met, val_res_all= self.metric_ftns({i: [gt] for i, gt in enumerate(val_gts)},
                                           {i: [re] for i, re in enumerate(val_res)})
                log.update(**{f'{split}_' + k: v for k, v in val_met.items()})
                if split != 'test':
                    val_res, val_gts = None, None
        # if split == 'test':
        #     torch.save([img_ids_list,selected_regions,att_recrod_preds],'test_sg_data.pth')


        return log, val_res, val_gts, img_ids, val_res_all

    def _pad(self, data):
        padded_data = torch.full((len(data), self.max_seq_length - 1), self.pad_idx, device=data[0].device)
        for i, cur_data in enumerate(data):
            padded_data[i, :cur_data.size(0)] = cur_data
        return padded_data

    def test(self, save_dir=''):
        self.logger.info('Starting evaluating the best checkpoint in test set.')
        log, val_res, val_gts, img_ids, val_res_all = self._valid(0, 'test')
        self.logger.info('The result for the best performed models in test set.')
        for key, value in log.items():
            self.logger.info('\t{:15s}: {}'.format(str(key), value))
        save_dir = self.checkpoint_dir if len(save_dir) == 0 else save_dir
        self.save_caption(val_res, val_res_all, val_gts, img_ids, log, save_dir=save_dir)

    def save_caption(self, val_res,val_res_all, val_gts, img_ids, log, save_dir=''):
        rank = torch.distributed.get_rank()
        val_res_all = [[val_res_all[m][i] for m in val_res_all.keys()] for i in range(len(img_ids))]
        data = (img_ids, val_res, val_gts, val_res_all)
        save_data = [{'img_id': img_id, 'pred': pred, 'gt': gt, 'score':val_res_cur} for img_id, pred, gt, val_res_cur in zip(*data)]
        save_data = [log] + save_data
        with open(f'caption_{rank}.json', 'w') as f:
            json.dump(save_data, f)

        save_path = save_dir
        torch.distributed.barrier()
        if rank == 0:
            jsons = list()
            paths = list(glob.glob("caption_*.json"))
            for path in paths:
                with open(path, "r") as fp:
                    jsons += json.load(fp)
            os.makedirs(save_path, exist_ok=True)
            with open(f"{save_path}/caption_data.json", "w") as fp:
                json.dump(jsons, fp, indent=4)
        torch.distributed.barrier()
        self.logger.info(f'Save prediction in directory {save_path}.')
        os.remove(f"caption_{rank}.json")

        return log

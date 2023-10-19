import os
from abc import abstractmethod

import time
import torch
import datetime
import pandas as pd
from numpy import inf
from tqdm import tqdm
from modules.utils import auto_resume_helper, get_grad_norm, reduce_tensor
from modules.weighted_mesloss import Weighted_MSELoss
from torch.cuda.amp import GradScaler, autocast
from timm.utils import AverageMeter
import torch.distributed as dist
from .utils import clip_grad_norm_, calculate_auc, get_region_mask
import numpy as np
from modules.beam_search import BeamSearch
from modules.loss import AsymmetricLoss,AsymmetricLossOptimized

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
        self.early_stop = config['early_stop']
        self.local_rank = config['local_rank']
        # if args.debug:
        #     for submodule in model.modules():
        #         submodule.register_forward_hook(nan_hook)
        self.model = model
        self.att_cls = config['att_cls']  # attribute classification
        # for name, module in self.model.named_modules():
        #     module.register_backward_hook(get_activations(name))
        self.optimizer = optimizer
        self.scaler = GradScaler(enabled=self.use_amp, init_scale=256)

        self.region_cls = config['region_cls']
        self.clip_option = config['clip_option']
        self.use_focal_ls = config['use_focal_ls']
        if self.region_cls:
            if self.use_focal_ls:
                self.region_cls_criterion = AsymmetricLossOptimized(gamma_neg=4, gamma_pos=1, clip=0.05, reduction='mean')
            else:
                self.region_cls_criterion = torch.nn.BCEWithLogitsLoss()
            self.region_cls_w = config['region_cls_w']

        if self.att_cls:
            if self.use_focal_ls:
                self.att_cls_criterion = AsymmetricLossOptimized(gamma_neg=4, gamma_pos=1, clip=0.05, reduction='mean')

            else:
                self.att_cls_criterion = torch.nn.BCEWithLogitsLoss()
            self.att_cls_w = config['att_cls_w']

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

        # keys to cuda
        self.keys = set(['image','text', 'mask', 'boxes', 'box_labels', 'box_masks', 'region_labels', 'attribute_labels',
                         'attribute_ids'])

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
    def test(self, epoch, split):
        raise NotImplementedError

    def train(self):

        not_improved_count = 0
        for epoch in range(self.start_epoch, self.epochs + 1):
            start = time.time()
            log = self._train_epoch(epoch)

            # evaluate model performance according to configured metric,
            log.update(self._valid(epoch, 'val'))
            log.update(self._valid(epoch, 'test'))
            # save logged informations into log dict
            # synchronize log in different gpu
            log = self._synchronize_data(log)

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
                self.logger.info(f'There are {zero_count} samples without any region selected in this epoch.' )

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
        state = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'lr_scheduler': self.lr_scheduler.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scaler': self.scaler.state_dict(),
            # 'amp':amp.state_dict() if self.amp_opt_level!='O0' else None,
            'monitor_best': self.mnt_val_best,
            'monitor_test_best': self.mnt_test_best,
            'best_epoch': self.best_epoch,
        }
        filename = os.path.join(self.checkpoint_dir, 'current_checkpoint.pth')
        # filename = os.path.join(self.checkpoint_dir, 'checkpoint'+str(epoch)+'.pth')
        torch.save(state, filename)
        print("Saving checkpoint: {} ...".format(filename))
        if save_best:
            best_path = os.path.join(self.checkpoint_dir, 'model_best.pth')
            torch.save(state, best_path)
            print("Saving current best: model_best.pth ...")

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
        region_cls_losses = AverageMeter()
        attribute_cls_losses = AverageMeter()
        norm_meter = AverageMeter()
        num_steps = len(self.train_dataloader)
        self.model.train()
        # cur_lr = [param_group['lr'] for param_group in self.optimizer.param_groups]
        self.optimizer.zero_grad()
        device = self.model.device
        with tqdm(desc='Epoch %d - train' % epoch, disable=dist.get_rank() != self.local_rank,
                  unit='it', total=len(self.train_dataloader)) as pbar:
            for batch_idx, data in enumerate(self.train_dataloader):
                batch_dict = {key: data[key].to(device, non_blocking=True) for key in data.keys() if key in self.keys}

                # images, reports_ids, reports_masks = data['image'].to(device, non_blocking=True), \
                #                                      data['text'].to(device, non_blocking=True), \
                #                                      data['mask'].to(device, non_blocking=True)
                # boxes, box_labels, region_labels, box_masks = None, None, None, None
                # if self.region_cls:
                #     boxes, box_labels, box_masks, region_labels = data['boxes'].to(device, non_blocking=True), \
                #                                                   data['box_labels'].to(device, non_blocking=True), \
                #                                                   data['box_masks'].to(device, non_blocking=True), \
                #                                                   data['region_labels'].to(device, non_blocking=True)
                #     if self.att_cls:
                #         attribute_labels = data['attribute_labels'].to(device, non_blocking=True)

                self.optimizer.zero_grad()
                with autocast(dtype=torch.float16):
                    # region logits is None when att_cls disabled
                    output = self.model(batch_dict,mode='train')
                    rrg_preds = output['rrg_preds']
                    loss = self.criterion(rrg_preds, batch_dict['text'], batch_dict['mask'])
                    ce_losses.update(loss.item())
                    if self.region_cls:
                        region_logits = output['region_logits']
                        region_cls_loss = self.region_cls_criterion(region_logits, batch_dict['region_labels'])
                        loss = loss + self.region_cls_w * region_cls_loss
                        region_cls_losses.update(region_cls_loss.item())

                    if self.att_cls:
                        attribute_logits = output['att_logits']
                        attribute_cls_loss = self.att_cls_criterion(attribute_logits, batch_dict['attribute_labels'])
                        loss = loss + self.att_cls_w * attribute_cls_loss
                        attribute_cls_losses.update(attribute_cls_loss.item())

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
                memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
                # cur_lr = [param_group['lr'] for param_group in self.optimizer.param_groups]
                pbar.set_postfix(ce=f'{ce_losses.val:.4f} ({ce_losses.avg:.4f})\t',
                                 rg_cls=f'{region_cls_losses.val:.4f} ({region_cls_losses.avg:.4f})\t',
                                 att_cls=f'{attribute_cls_losses.val:.4f} ({attribute_cls_losses.avg:.4f})\t',
                                 mem=f'mem {memory_used:.0f}MB',
                                 norm=f'{norm_meter.val:.4f} ({norm_meter.avg:.4f})')
                pbar.update()
            # if self.early_exit and batch_idx>100:
            #     torch.save(self.model.records, 'cam_records_fblrelu.pth')
            #     exit()

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
        val_att_cls_losses = AverageMeter()
        log = {}
        self.model.eval()
        dataloader = self.val_dataloader if split == 'val' else self.test_dataloader
        device = self.model.device
        region_preds, region_targets = [], []
        # attribute_preds, attribute_targets = [], []

        with tqdm(desc=f'Epoch %d - {split}' % epoch, unit='it', total=len(dataloader),
                  disable=dist.get_rank() != self.local_rank) as pbar:
            with torch.no_grad():
                val_gts, val_res = [], []
                for batch_idx, data in enumerate(dataloader):
                    # images, reports_ids, reports_masks = data['image'].to(device, non_blocking=True), \
                    #                                      data['text'].to(device, non_blocking=True), \
                    #                                      data['mask'].to(device, non_blocking=True)
                    # total_attn, boxes, return_feats, box_labels = None, None, None, None
                    batch_dict = {key:data[key].to(device, non_blocking=True) for key in data.keys() if key in self.keys}
                    if self.region_cls:
                        # boxes, box_labels, region_labels = data['boxes'].to(device, non_blocking=True), \
                        #                                    data['box_labels'].to(device, non_blocking=True), \
                        #                                    data['region_labels'].to(device, non_blocking=True)
                        region_labels = batch_dict['region_labels']
                        region_masks = get_region_mask(region_labels).cpu()
                        region_labels = region_labels[region_masks]

                        # if self.att_cls:
                        #     attribute_masks = data['attribute_masks']
                        #     attribute_labels = data['attribute_labels']
                        #     if len(attribute_labels) > 0:
                        #         attribute_labels = data['attribute_labels'].to(device, non_blocking=True)

                    with autocast(dtype=torch.float16):
                        # if split == 'val':
                        #     output = self.model(batch_dict,mode='train', return_feats=True)
                        #     rrg_preds = output['rrg_preds']
                        #     loss = self.criterion(rrg_preds, batch_dict['text'], batch_dict['mask'])
                        #     val_ce_losses.update(loss.item(),rrg_preds.size(0))
                        # # output, _ = self.model(images,reports_ids,mode='sample')
                        # else:
                        output = self.model(batch_dict,mode='sample')

                        patch_feats = output['encoded_img_feats']

                        if self.region_cls:
                            region_logits = output['region_logits'][region_masks]
                            region_probs = output['region_probs'][region_masks]
                            if len(region_labels) > 0:
                                val_region_cls_loss = self.region_cls_criterion(region_logits, region_labels)
                                val_region_cls_losses.update(val_region_cls_loss.item())
                                if len(region_preds) > 0:
                                    region_preds = torch.cat((region_preds, region_probs.cpu()), dim=0)
                                    region_targets = torch.cat((region_targets, region_labels.cpu()), dim=0)
                                else:
                                    region_preds = region_probs.cpu()
                                    region_targets = region_labels.cpu()

                        # if self.att_cls:
                        #     att_logits = output['att_logits'][attribute_masks == 1]
                        #     att_probs = output['att_probs'][attribute_masks == 1]
                        #     if len(attribute_labels) > 0:
                        #         val_att_cls_loss = self.att_cls_criterion(att_logits, attribute_labels)
                        #         val_att_cls_losses.update(val_att_cls_loss.item(), sum(attribute_masks))
                        #         if len(attribute_preds) > 0:
                        #             attribute_preds = torch.cat((attribute_preds, att_probs.cpu()), dim=0)
                        #             attribute_targets = torch.cat((attribute_targets, attribute_labels.cpu()), dim=0)
                        #     else:
                        #         if len(attribute_labels) > 0:
                        #             attribute_preds = attribute_preds.cpu()
                        #             attribute_targets = attribute_targets.cpu()
                        if self.use_new_bs:
                            output = self.beam_search.caption_test_step(self.model.module, image_embeds=patch_feats)
                        else:
                            output = self.beam_search.sample(self.model.module, patch_feats=patch_feats)

                    reports = self.tokenizer.decode_batch(output['preds'].numpy())
                    ground_truths = self.tokenizer.decode_batch(batch_dict['text'][:, 1:].cpu().numpy())
                    val_res.extend(reports)
                    val_gts.extend(ground_truths)
                    memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
                    # pbar.set_postfix(ce_ls=val_ce_losses / (batch_idx + 1), cls_ls=val_img_cls_losses / (batch_idx + 1),
                    #                  mse_ls=val_mse_losses / (batch_idx + 1), mem=f'mem {memory_used:.0f}MB')
                    pbar.update()
                val_met = self.metric_ftns({i: [gt] for i, gt in enumerate(val_gts)},
                                           {i: [re] for i, re in enumerate(val_res)})
                log.update(**{f'{split}_' + k: v for k, v in val_met.items()})
                if split == 'val':
                    log.update({'val_ce_loss': val_ce_losses.avg})
                if self.region_cls:
                    region_auc = calculate_auc(preds=region_preds, targets=region_targets)
                    log.update({f'{split}_region_auc': region_auc, f"{split}_rg_loss": val_region_cls_losses.avg})
                # if self.att_cls:
                #     att_auc = calculate_auc(preds=attribute_preds, targets=attribute_targets)
                #     log.update({f'{split}_att_auc': region_auc, f"{split}_att_loss": val_att_cls_losses.avg})
        return log

    def test(self, epoch, split):
        self.logger('Starting evaluating the best checkpoint in test set.')
        self._valid(epoch,'test')

import torch
import spacy
import numpy as np
import os
import random
from modules.tokenizers_origin import Tokenizer
from modules.rule_tokenizer import RuleTokenizer
from data.dataloaders import R2DataLoader
from modules.metrics import compute_scores, CaptionScorer
from modules.optimizers import build_optimizer, build_lr_scheduler
from modules.trainer import Trainer
from modules.loss import compute_loss
from models.origin_model import RRGModel
from modules.utils import auto_resume_helper, load_checkpoint
from modules.logger import create_logger
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
import torch.backends.cudnn as cudnn
from config import ex
# from data.multitask_datamodule import MTDataModule
import resource
import copy

os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'
import time
# from apex import amp
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
import os

import datetime


def setup(world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12345"
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = world_size

def scale_lr(config):
    # linear scale the learning rate according to total batch size, may not be optimal
    linear_scaled_base = config['lr_base'] * config['batch_size'] * dist.get_world_size() / 64.0

    # linear_scaled_crosslr = config['cross_base_lr'] * config['per_gpu_batchsize'] * dist.get_world_size() / 64.0
    linear_scaled_warmup_lr = linear_scaled_base / config['warmup_ratio']
    linear_scaled_min_lr = linear_scaled_warmup_lr
    # gradient accumulation also need to scale the learning rate
    config['lr_base'] = linear_scaled_base
    config['warmup_lr'] = linear_scaled_warmup_lr
    config['min_lr'] = linear_scaled_min_lr


@ex.automain
def main(_config):
    _config = copy.deepcopy(_config)
    world_size = _config["n_gpu"]
    # setup(str(_config["n_gpu"]))

    torch.cuda.set_device(_config['local_rank'])
    torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size)
    rank = dist.get_rank()
    device_id = rank % torch.cuda.device_count()
    # torch.distributed.barrier()

    seed = _config['seed'] + dist.get_rank()
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = False
    save_dir = os.path.join(_config['output'], _config['dataset_name'], _config['exp_name'])
    os.makedirs(save_dir, exist_ok=True)
    logger = create_logger(output_dir=save_dir, dist_rank=_config['local_rank'], name=_config['exp_name'])

    writer = SummaryWriter(log_dir=os.path.join(_config['output'], _config['dataset_name'], _config['exp_name']))
    # create tokenizer

    # create data loader
    train_dataloader = R2DataLoader(_config, None, split='train', shuffle=True)
    tokenizer = train_dataloader.dataset.tokenizer  # remember to delete the old vocab when new one
    all_texts = train_dataloader.dataset.all_texts
    _config['vocab_size'] = tokenizer.get_vocab_size()

    val_dataloader = R2DataLoader(_config, tokenizer, split='val', shuffle=False)
    test_dataloader = R2DataLoader(_config, tokenizer, split='test', shuffle=False)


    dm = {'train_dataloader': train_dataloader, 'val_dataloader': val_dataloader, 'test_dataloader': test_dataloader,
          'tokenizer': tokenizer}
    # dm = MTDataModule(_config, dist=True)

    # build model architecture
    model = RRGModel(tokenizer, logger, _config)

    if _config['scale_lr']:
        scale_lr(_config)

    optimizer = build_optimizer(model, _config)

    logger.info(model)

    resume = False

    model = RRGModel(tokenizer, logger, _config)
    state_dict = torch.load(_config['load_path'])['state_dict']
    model.load_state_dict(state_dict)
    model = model.to(device_id)

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device_id], broadcast_buffers=False,
                                                      find_unused_parameters=True)
    model_without_ddp = model.module

    if _config['compile']:
        model = torch.compile(model)

    # print(model)

    lr_scheduler = build_lr_scheduler(_config, optimizer, len(train_dataloader))

    if dist.get_rank() == _config['local_rank']:
        logger.info(_config)
        logger.info(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
        # path = os.path.join(save_dir, _config['exp_name'], "config.json")
        # with open(path, "w") as f:
        #     f.write(_config.dump())
        # logger.info(f"Full config saved to {path}")
        # logger.info(_config.dump())
        n_parameters = sum(p.numel() for p in model_without_ddp.parameters() if p.requires_grad)
        logger.info(f"number of params: {n_parameters}")
        if hasattr(model_without_ddp, 'flops'):
            flops = model_without_ddp.flops()
            logger.info(f"number of GFLOPs: {flops / 1e9}")


    # get function handles of loss and metrics
    criterion = compute_loss
    metrics = CaptionScorer(all_texts)

    # build optimizer, learning rate scheduler

    # build trainer and start to train
    trainer = Trainer(model, criterion, metrics, optimizer, lr_scheduler, dm, writer, logger, _config)
    save_dir = os.path.join(*_config['load_path'].split('/')[:-1])
    trainer.test(save_dir=save_dir)

    # if _config["test_after"]:
    #
    #     # if args.amp_opt_level != "O0":
    #     #     model, optimizer = amp.initialize(model, optimizer, opt_level=args.amp_opt_level)
    #
    #     model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device_id], broadcast_buffers=False,
    #                                                       find_unused_parameters=_config['debug'])
    #     logger.info(f'loading model weights from {load_path}.')
    #     trainer = Trainer(model, criterion, metrics, optimizer, lr_scheduler, dm, writer, logger, _config)
    #     trainer.test()
    writer.close()




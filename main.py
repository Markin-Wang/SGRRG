import torch
import spacy
import numpy as np
import os
import random
from modules.tokenizers_origin import Tokenizer
from modules.rule_tokenizer import RuleTokenizer
from data.dataloaders import R2DataLoader
from modules.metrics import compute_scores
from modules.optimizers import build_optimizer, build_lr_scheduler
from modules.trainer import Trainer
from modules.loss import compute_loss
from models.origin_model import RRGModel
from modules.utils import parse_args, auto_resume_helper, load_checkpoint
from modules.logger import create_logger
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
import torch.backends.cudnn as cudnn
from config import ex
# from data.multitask_datamodule import MTDataModule
import resource
import copy
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (20480, rlimit[1]))
os.environ["NCCL_DEBUG"] = "INFO"
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'
import time
#from apex import amp
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import os

import datetime


def setup(world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12345"
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = world_size

@ex.automain
def main(_config):
    _config = copy.deepcopy(_config)
    world_size = _config["n_gpu"]
    #setup(str(_config["n_gpu"]))

    torch.cuda.set_device(_config['local_rank'])
    torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size)
    rank = dist.get_rank()
    device_id = rank % torch.cuda.device_count()
    #torch.distributed.barrier()

    seed = _config['seed'] + dist.get_rank()
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = False
    save_dir = os.path.join(_config['output'],_config['dataset_name'])
    os.makedirs(save_dir, exist_ok=True)
    logger = create_logger(output_dir=save_dir,dist_rank=_config['local_rank'], name=_config['exp_name'])



    writer = SummaryWriter(log_dir=os.path.join(_config['output'], _config['exp_name']))
    # create tokenizer


    # create data loader
    train_dataloader = R2DataLoader(_config, None, split='train', shuffle=True)
    tokenizer = train_dataloader.dataset.tokenizer # remember to delete the old vocab when new one
    _config['vocab_size'] = tokenizer.get_vocab_size()

    val_dataloader = R2DataLoader(_config, tokenizer, split='val', shuffle=False)
    test_dataloader = R2DataLoader(_config, tokenizer, split='test', shuffle=False)

    # starttime = datetime.datetime.now()
    # for i in range(10):
    #     for data in train_dataloader:
    #         pass
    # endtime = datetime.datetime.now()
    # print(111111,(endtime-starttime).seconds)
    # exit(0)

    # 35s for iu xray loading 5 times array
    # 34s for iu xray loading 5 time origin


    dm = {'train_dataloader':train_dataloader,'val_dataloader':val_dataloader,'test_dataloader':test_dataloader,'tokenizer':tokenizer}
    # dm = MTDataModule(_config, dist=True)


    # build model architecture
    model = RRGModel(tokenizer, logger, _config)

    optimizer = build_optimizer(model, _config)

    logger.info(model)

    resume = False

    model = model.to(device_id)
    # if args.amp_opt_level != "O0":
    #     model, optimizer = amp.initialize(model, optimizer, opt_level=args.amp_opt_level)

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device_id], broadcast_buffers=False,find_unused_parameters=True)
    model_without_ddp = model.module


    if config['compile']:
        model = torch.compile(model,mode='reduce-overhead')

    #print(model)

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

    if len(_config['load_path'])>0:
        state_dict = torch.load(args.pretrained)['model']
        logger.info(state_dict.keys())
        state_dict.pop('head.weight')
        state_dict.pop('head.bias')
        model.load_state_dict(state_dict, strict=False)
        logger.info(f'loading pretrained model {_config["load_path"]}, ignoring auto resume')

    # get function handles of loss and metrics
    criterion = compute_loss
    metrics = compute_scores

    # build optimizer, learning rate scheduler

    # build trainer and start to train
    trainer = Trainer(model, criterion, metrics, optimizer, lr_scheduler, dm, writer, logger, _config)
    trainer.train()
    writer.close()



def scale_lr(config):
    #linear scale the learning rate according to total batch size, may not be optimal
    linear_scaled_imglr = config['lr_ve'] * config['batch_size'] * dist.get_world_size() / 64.0
    linear_scaled_textlr = config['lr_ed'] * config['batch_size'] * dist.get_world_size() / 64.0
    # linear_scaled_crosslr = config['cross_base_lr'] * config['per_gpu_batchsize'] * dist.get_world_size() / 64.0
    linear_scaled_warmup_lr = linear_scaled_imglr / config['warmup_ratio']
    linear_scaled_min_lr = linear_scaled_warmup_lr
    # gradient accumulation also need to scale the learning rate
    config['lr_ve'] = linear_scaled_imglr
    config['lr_ed'] = linear_scaled_imglr
    config['warmup_lr'] = linear_scaled_warmup_lr
    config['min_lr'] = linear_scaled_warmup_lr

import torch
import argparse
import numpy as np
import os
import random
from modules.tokenizers_origin import Tokenizer
from modules.dataloaders import R2DataLoader
from modules.metrics import compute_scores
from modules.optimizers import build_optimizer, build_lr_scheduler
from modules.trainer import Trainer
from modules.loss import compute_loss
from modules.utils import parse_args, auto_resume_helper, load_checkpoint
from modules.logger import create_logger
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
from config_swin import get_config
from tqdm import tqdm
import json
import torch.backends.cudnn as cudnn
from models.model import RRGModel
from copy import deepcopy as dc


def test(args, config):
    # parse arguments
    args, config = parse_args()
    print(args)
    # create tokenizer
    tokenizer = Tokenizer(args)

    # create data loader
    train_dataloader = R2DataLoader(args, tokenizer, split='train', shuffle=True)
    val_dataloader = R2DataLoader(args, tokenizer, split='val', shuffle=False)
    test_dataloader = R2DataLoader(args, tokenizer, split='test', shuffle=False)

    # build model architecture
    model = RRGModel(args, tokenizer, logger, config)



    model.cuda()
    # if args.amp_opt_level != "O0":
    #     model, optimizer = amp.initialize(model, optimizer, opt_level=args.amp_opt_level)

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.LOCAL_RANK],
                                                      output_device=config.LOCAL_RANK, broadcast_buffers=False)

    if args.model_path:
        state_dict = torch.load(args.model_path)['state_dict']
        logger.info(state_dict.keys())
        model.load_state_dict(state_dict)
        logger.info(f'loading pretrained model {args.model_path}, ignoring auto resume')
    #model_without_ddp = model.module

    metrics = compute_scores

    model = model.cuda()

    model.eval()
    data = []
    with torch.no_grad():
        records = []
        test_gts, test_res = [], []
        for batch_idx, (images_id, images, reports_ids, reports_masks, labels) in enumerate(tqdm(test_dataloader)):
            vis_data = {}
            vis_data['id'] = images_id
            vis_data['img'] = images
            vis_data['labels'] = labels
            #
            # print(images_id)

            images, reports_ids, reports_masks, labels = images.cuda(), reports_ids.cuda(), \
                                                         reports_masks.cuda(), labels.cuda()
            #if images_id[0] != 'data/mimic_cxr/images/p10/p10402372/s51966612/8797515b-595dfac0-77013a06-226b52bd-65681bf2.jpg':
            #    continue
            #print('000', reports_ids, reports_ids.shape)
            output, attns = model(images, labels=labels, mode='sample')
            if args.addcls:
                _, logits, cams, fore_map, total_attns, idxs, align_attns_train =  model(images, reports_ids, labels, mode='train')
                vis_data['cams'] = cams.detach().cpu()
                vis_data['logits'] = logits.detach().cpu()
                if fore_map is not None:
                    vis_data['fore_map'] = fore_map.detach().cpu()
                if total_attns is not None:
                    vis_data['total_attns'] = total_attns
                if align_attns_train is not None:
                    vis_data['align_attns_train'] = align_attns_train
            else:
                idxs = None


            vis_data['attn'] = attns




            # if args.n_gpu > 1:
            #     reports = model.module.tokenizer.decode_batch(output.cpu().numpy())
            #     ground_truths = model.module.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())
            # else:
            #     reports = model.tokenizer.decode_batch(output.cpu().numpy())
            #     ground_truths = model.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())

            vis_data['pre'] = []
            vis_data['gt'] = []
            vis_data['met'] = []
            vis_data['train_pre'] = []
            if idxs is not None:
                vis_data['idxs'] = idxs
            # print(reports_ids[0].shape, len(ground_truths[0]))


            for id, out, report_id in zip(images_id, output, reports_ids):
                predict = tokenizer.decode(out.cpu().numpy())
                gt = tokenizer.decode(report_id[1:].cpu().numpy())
                val_met = metrics({id: [gt]}, {id: [predict]})
                # vis_data['pre'].append(out.cpu().numpy())
                # vis_data['gt'].append(report_id[1:].cpu().numpy())
                vis_data['pre'].append(predict)
                vis_data['gt'].append(gt)
                # if val_met['BLEU_4'] > 0.3:
                #     records.append({'id':id,'gt':gt,'pre':predict})
                vis_data['met'].append(val_met)
            data.append(vis_data)

            reports = tokenizer.decode_batch(output.cpu().numpy())
            ground_truths = tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())
            test_res.extend(reports)
            test_gts.extend(ground_truths)
        #vis_data['records'] = records
        test_met = metrics({i: [gt] for i, gt in enumerate(test_gts)},
                                   {i: [re] for i, re in enumerate(test_res)})

    print(test_met)

    # f = open('mimic_prediction_our03.json', 'w', encoding='utf-8')
    # json.dump(records, f, indent=1)
    # f.close()
    data = cat_data(data)
    torch.save(data, os.path.join('mic_sts_st_2e-6_9e-5_1e2_wd5e-2_4_bs192_3gpu_ep40_et.pth'))
    #torch.save([tokenizer.idx2token, tokenizer.token2idx], os.path.join('visualizations','vis', args.dataset_name+'token_map.pth'))


def cat_data(data):
    temp = dc(data[0])
    for i in range(1,len(data)):
        temp['id'].extend(data[i]['id'])
        temp['img'] = torch.cat((temp['img'], data[i]['img']))
        temp['labels'] = torch.cat((temp['labels'], data[i]['labels']))
        temp['pre'].extend(data[i]['pre'])
        temp['train_pre'].extend(data[i]['train_pre'])
        temp['attn'].extend(data[i]['attn'])
        temp['gt'].extend(data[i]['gt'])
        temp['met'].extend(data[i]['met'])
        saved_data = [{'id': temp['id'][i], 'label': temp['labels'][i], 'pre': temp['pre'][i], 'gt': temp['gt'][i],
                       'met': temp['met'][i]}
                      for i in range(len(temp['id']))]
    return saved_data




if __name__ == '__main__':
    args, config = parse_args()

    if 'RANK' in os.environ:
        rank = int(os.environ["RANK"])
    else:
        rank = -1

    if 'WORLD_SIZE' in os.environ:
        world_size = int(os.environ['WORLD_SIZE'])
    else:
        world_size = -1

    ngpus_per_node = torch.cuda.device_count()

    if config.LOCAL_RANK != -1:  # for torch.distributed.launch
        args.local_rank = config.LOCAL_RANK
        args.rank = config.LOCAL_RANK

    torch.cuda.set_device(config.LOCAL_RANK)
    torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    torch.distributed.barrier()

    seed = config.SEED + dist.get_rank()
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = False

    logger = create_logger(output_dir=config.OUTPUT, dist_rank=config.LOCAL_RANK, name=f"{config.MODEL.NAME}")
    # print config
    test(args, config)

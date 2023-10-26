from sacred import Experiment
import numpy as np

ex = Experiment("FIBER")


def _loss_names(d):
    ret = {
        "itm": 0,
        "itc": 0,
        "mlm": 0,
        "vqa": 0,
        "nlvr2": 0,
        "caption_mle": 0,
        "caption_gold": 0,
        "caption_cider": 0,
    }
    ret.update(d)
    return ret


@ex.config
def config():
    exp_name = "fiber"
    seed = 0
    data_dir = 'datasets'
    # datasets = ["coco", "vg", "sbu", "gcc"]
    loss_names = _loss_names({"itm": 1, "mlm": 1, "itc": 1})
    # batch_size = (
    #     4096  # this is a desired batch size; pl trainer will accumulate gradients when per step batch is smaller.
    # )
    batch_size = 16
    eval_batch_size = batch_size
    dataset_name = 'mimic_cxr'
    label_path = 'labels/labels.json'
    pretrained = ''
    model_path = ''
    output = 'results'  # output dir
    eval = False  # whether to perform evaluation only
    img_backbone = 'resnet101'

    logs_folder = 'tensorboard_logs'
    max_seq_length = 60
    threshold = 10
    num_workers = 8

    # Model
    use_dropout = True
    use_ln = False
    ve_name = 'swin_s'
    ed_name = 'st_trans'
    visual_extractor_pretrained = True  # whether to load the pretrained visual extractor
    d_model = 512  # the dimension of Transformer.
    d_ff = 512  # the dimension of Transformer.
    d_vf = 2048  # the dimension of the patch features.
    num_heads = 8
    num_layers_en = 3
    num_layers_de = 3
    dropout = 0.1
    logit_layers = 1
    bos_idx = 0
    eos_idx = 0
    pad_idx = 0
    use_bn = 0
    drop_prob_lm = 0.5

    # Beam Search
    sample_method = 'beam_search'
    use_new_bs = False
    beam_size = 3
    temperature = 1.0
    sample_n = 1
    group_size = 1
    diversity_lambda = 0.5
    length_penalty = ''
    suppress_UNK = 0

    # learning schedule
    scale_lr = False
    warmup_ratio = 10
    warmup_epochs = 5
    output_logsoftmax = 1
    decoding_constraint = 0
    block_trigrams = 1
    optim = 'AdamW'
    lr_base = 5e-5
    mul_new = 2  # lr_base * mul_new for new parameters
    weight_decay = 5e-5
    decay_epochs = 10
    amsgrad = False
    lr_scheduler = 'step'
    step_size = 50
    decay_rate = 0.8
    clip_value = 0.1
    clip_option = 'value'
    eps = 1e-8
    warmup_ratio = 100
    msw_w = 0.5

    # trainer setting
    image_size = 224
    rotate_degree = 10
    compile = False  # whether use the torch.compile for model, only supported by pytorch >= 2.0
    dsr = 1  # down sample rate for the dataset
    resolution_before = 224
    n_gpu = 1
    epochs = 100
    use_amp = True
    save_dir = 'results'
    record_dir = 'records/'

    # attribute classification setting
    att_cls = False  # attribute classification
    att_cls_w = 0.5
    feature_size = image_size // 32  # 32 is the downsampling rate for the visual extractor
    output_size = 1
    num_classes = 29  # the number of anatomical locations
    region_select_threshold = 0.5  # the threshold used to select the region after sigmoid
    att_select_threshold = 0.5
    att_pad_idx = -10000
    region_cls_only = False  # perform region classification in visual extractor
    region_cls = False
    region_cls_w = 0.5
    num_attributes = 849  # for one head attribute classification 848 for mimic-split
    use_box_feats = False

    # scene graph setting
    use_sg = False
    sgave = False # scene graph aided vision encoder
    sgade = False # scene graph aided decoder
    num_layers_sgen = 3
    use_region_type_embed = False
    use_focal_ls = False
    encode_type = 'oa-c'
    # object-attribute coupled (oa-c): node-att to node-att
    # object-attribute decomposed (oa-d): node to node-att then nodes to nodes
    # object-attribute decomposed completely (oa-dc): node to att then nodes to nodes

    #
    save_period = 1
    monitor_mode = 'max'
    early_stop = 8
    fp16 = True
    balanced = False
    vis = False
    test = False
    debug = False
    num_patches = 98  # the number of image patches in encoder
    block3 = False
    encode_text = False
    num_layers_ten = 0

    cfg = 'configs/swin_tiny_patch4_window7_224.yaml'
    local_rank = 0
    load_path = ''

    # Evaluation
    monitor_metric = 'BLEU_4'

    test_after = False
    start_eval = 0  # the epoch starts to perform evaluation
    resume = False
    early_exit = False  # used for test
    layer_id = 2  # the layer id in encoder to select attention
    pe = 'none'  # whether to use absolute position embedding in encoder


@ex.named_config
def task_train_caption_iu():
    exp_name = 'iu_test'
    dataset_name = 'iu_xray'
    max_seq_length = 60
    threshold = 3
    batch_size = 8
    epochs = 40
    lr_base = 1e-3
    mul_new = 2
    img_backbone = 'swin_base_patch4_window7_224_in22k'
    d_vf = 1024
    ed_name = 'st_trans'
    seed = 9223
    use_amp = True


@ex.named_config
def task_train_caption_mimic():
    exp_name = 'mimic_test'
    dataset_name = 'mimic_cxr'
    dsr = 2
    max_seq_length = 100
    threshold = 10
    batch_size = 16
    epochs = 30
    lr_base = 5e-5
    mul_new = 2
    img_backbone = 'swin_base_patch4_window7_224_in22k'
    resolution_before = 224
    image_size = 224
    d_vf = 1024
    ed_name = 'st_trans'
    seed = 9223
    use_amp = True


@ex.named_config
def task_train_caption_cxr_gnome():
    exp_name = 'cxr_gnome_test'
    dataset_name = 'cxr_gnome'
    dsr = 1
    max_seq_length = 100
    threshold = 10
    batch_size = 16
    epochs = 30
    lr_base = 5e-5
    mul_new = 2
    img_backbone = 'swin_base_patch4_window7_224_in22k'
    resolution_before = 224
    image_size = 224
    d_vf = 1024
    ed_name = 'st_trans'
    seed = 9223
    use_amp = True


# for mimic-cxr attribute classification
id2cat = [('left lung', 68),
          ('right lung', 68),
          ('cardiac silhouette', 34),
          ('mediastinum', 40),
          ('left lower lung zone', 57),
          ('right lower lung zone', 58),
          ('right hilar structures', 48),
          ('left hilar structures', 49),
          ('upper mediastinum', 30),
          ('left costophrenic angle', 34),
          ('right costophrenic angle', 34),
          ('left mid lung zone', 48),
          ('right mid lung zone', 48),
          ('aortic arch', 12),
          ('right upper lung zone', 49),
          ('left upper lung zone', 45),
          ('right hemidiaphragm', 14),
          ('right clavicle', 16),
          ('left clavicle', 15),
          ('left hemidiaphragm', 13),
          ('right apical zone', 24),
          ('trachea', 10),
          ('left apical zone', 23),
          ('carina', 5),
          ('svc', 9),
          ('right atrium', 7),
          ('cavoatrial junction', 4),
          ('abdomen', 9),
          ('spine', 13)]

# generated from attribute annotation
catid2attrange = np.array(
    [[102, 169],
     [0, 67],
     [274, 307],
     [204, 243],
     [500, 556],
     [660, 717],
     [332, 379],
     [403, 451],
     [244, 273],
     [170, 203],
     [68, 101],
     [452, 499],
     [612, 659],
     [734, 745],
     [776, 824],
     [839, 883],
     [576, 589],
     [718, 733],
     [557, 571],
     [756, 768],
     [308, 331],
     [746, 755],
     [380, 402],
     [825, 829],
     [830, 838],
     [769, 775],
     [572, 575],
     [590, 598],
     [599, 611]]
)

# 849 attribute in total
cgnome_id2cat = np.array([67, 67, 33, 39, 57, 56, 46, 46, 28, 32, 32, 45, 47, 12, 46, 43, 13,
                       15, 15, 12, 22, 10, 21, 4, 9, 7, 4, 9, 12])

cgnome_cumcat = np.array(
    [0, 67, 134, 167, 206, 263, 319, 365, 411, 439, 471, 503, 548, 595, 607, 653, 696, 709, 724, 739, 751, 773, 783,
     804, 808, 817, 824, 828, 837])

cgnome_catid2attrange = np.array([[0, 66],
                                  [67, 133],
                                  [134, 166],
                                  [167, 205],
                                  [206, 262],
                                  [263, 318],
                                  [319, 364],
                                  [365, 410],
                                  [411, 438],
                                  [439, 470],
                                  [471, 502],
                                  [503, 547],
                                  [548, 594],
                                  [595, 606],
                                  [607, 652],
                                  [653, 695],
                                  [696, 708],
                                  [709, 723],
                                  [724, 738],
                                  [739, 750],
                                  [751, 772],
                                  [773, 782],
                                  [783, 803],
                                  [804, 807],
                                  [808, 816],
                                  [817, 823],
                                  [824, 827],
                                  [828, 836],
                                  [837, 848]])

categories =['left lung','right lung','cardiac silhouette','mediastinum','left lower lung zone','right lower lung zone',
             'right hilar structures','left hilar structures','upper mediastinum',
             'left costophrenic angle','right costophrenic angle',
             'left mid lung zone','right mid lung zone','aortic arch','right upper lung zone',
             'left upper lung zone','right hemidiaphragm','right clavicle','left clavicle',
             'left hemidiaphragm','right apical zone','trachea','left apical zone','carina',
             'svc','right atrium','cavoatrial junction','abdomen','spine',]
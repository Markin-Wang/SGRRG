import json
import os
import pandas as pd
import pyarrow as pa
import random
import json
from tqdm import tqdm
from glob import glob
from collections import defaultdict
import sys
import re
import resource
import pickle
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (20480, rlimit[1]))
print(rlimit)

def clean_report_mimic_cxr(report):
    report_cleaner = lambda t: t.replace('\n', ' ').replace('__', '_').replace('__', '_').replace('__', '_') \
        .replace('__', '_').replace('__', '_').replace('__', '_').replace('__', '_').replace('  ', ' ') \
        .replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ') \
        .replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.') \
        .replace('..', '.').replace('..', '.').replace('..', '.').replace('1. ', '').replace('. 2. ', '. ') \
        .replace('. 3. ', '. ').replace('. 4. ', '. ').replace('. 5. ', '. ').replace(' 2. ', '. ') \
        .replace(' 3. ', '. ').replace(' 4. ', '. ').replace(' 5. ', '. ') \
        .strip().lower().split('. ')
    sent_cleaner = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '')
                                    .replace('\\', '').replace("'", '').strip().lower())
    tokens = [sent_cleaner(sent) for sent in report_cleaner(report) if sent_cleaner(sent) != []]
    report = ' . '.join(tokens) + ' .'
    return report

def path2rest_mimic_cxr(id, iid2captions, iid2imgs, iid2split,data_root):
    path = os.path.join(data_root,'images',iid2imgs[id][0])
    with open(path, "rb") as fp:
        binary = fp.read()
    captions = iid2captions[id]
    captions = [clean_report_mimic_cxr(caption) for caption in captions]
    # if len(captions[0]) <=15 or captions[0] == 'see impression below .':
    #     print(captions)
    #     return None

    split = iid2split[id]
    return [binary, captions, id, split]


def make_arrow_mimic_cxr(data, dataset_name, data_root, save_dir):
    print(f"+ Pre-processing {dataset_name}...")
    iid2captions = defaultdict(list)
    iid2imgs = defaultdict(list)
    iid2split = dict()

    for split, split_data in data.items():
        if split != 'train': continue
        for sample in split_data:
            iid2captions[sample["id"]].extend([sample["report"]])
            iid2imgs[sample["id"]].extend(sample["image_path"])
            iid2split[sample["id"]] = split

    num_samples = len(iid2captions)
    img_paths = [path for v in tqdm(iid2imgs.values()) for path in v if os.path.exists(os.path.join(data_root,'images',path))]
    #img_paths = [path for v in tqdm(iid2imgs.values()) for path in v]
    print(f"+ {len(img_paths)} images / {num_samples} annotations")
    #statistics(iid2captions, iid2split)
    bs = [path2rest_mimic_cxr(id, iid2captions, iid2imgs, iid2split, data_root) for id in tqdm(iid2captions)]
    print('clean before:', len(bs))
    bs = [sample for sample in bs if sample is not None]
    print('clean after', len(bs))

    #for split in ["train", "val", "test"]:
    for split in ["train"]:
        bs = [b for b in bs if b[-1] == split]
        #dataframe = pd.DataFrame(batches, columns=["image", "caption", "image_id", "chexpert", "split"])
        table = pd.DataFrame(bs, columns=["image", "caption", "image_id", "split"])
        bs = None
        table = pa.Table.from_pandas(table)

        os.makedirs(save_dir, exist_ok=True)
        save_file = f"{save_dir}/{dataset_name}_{split}.arrow"
        print(f'Writing {save_file}...')
        with pa.OSFile(save_file, "wb") as sink:
            with pa.RecordBatchFileWriter(sink, table.schema) as writer:
                writer.write_table(table)


if __name__ == '__main__':
    data_root = sys.argv[1] + '/cxr_gnome'
    save_dir = sys.argv[2]
    ann_path = os.path.join(data_root,'annotations','report_ann.json')
    ann = json.loads(open(ann_path, 'r').read())
    print(1111,len(ann['train']),len(ann['val']),len(ann['test']))
    make_arrow_mimic_cxr(ann,'cxr_gnome', data_root, save_dir)
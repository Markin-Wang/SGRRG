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


def clean_report_iu_xray(report):
    report_cleaner = lambda t: t.replace('..', '.').replace('..', '.').replace('..', '.').replace('1. ', '') \
        .replace('. 2. ', '. ').replace('. 3. ', '. ').replace('. 4. ', '. ').replace('. 5. ', '. ') \
        .replace(' 2. ', '. ').replace(' 3. ', '. ').replace(' 4. ', '. ').replace(' 5. ', '. ') \
        .strip().lower().split('. ')
    sent_cleaner = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '').
                                    replace('\\', '').replace("'", '').strip().lower())
    tokens = [sent_cleaner(sent) for sent in report_cleaner(report) if sent_cleaner(sent) != []]
    report = ' . '.join(tokens) + ' .'
    return report


# def path2rest_iu_xray(id, iid2captions, iid2imgs, iid2chexpert, iid2split,data_root):
#     path1 = os.path.join(data_root,'images',iid2imgs[id][0])
#     path2 = os.path.join(data_root, 'images', iid2imgs[id][1])
#     with open(path1, "rb") as fp:
#         binary1 = fp.read()
#     with open(path2, "rb") as fp:
#         binary2 = fp.read()
#     captions = iid2captions[id]
#     captions = [clean_report_iu_xray(caption) for caption in captions]
#     # if len(captions[0]) <=15 or captions[0] == 'see impression below .':
#     #     print(captions)
#     #     return None
#     chexpert = iid2chexpert[id]
#     split = iid2split[id]
#     return [binary1, binary2, captions, id, chexpert, split]

def path2rest_iu_xray(id, iid2captions, iid2imgs, iid2chexpert, iid2split,data_root):
    path1 = os.path.join(data_root,'images',iid2imgs[id][0])
    path2 = os.path.join(data_root, 'images', iid2imgs[id][1])
    with open(path1, "rb") as fp:
        binary1 = fp.read()
    with open(path2, "rb") as fp:
        binary2 = fp.read()
    captions = iid2captions[id]
    captions = [clean_report_iu_xray(caption) for caption in captions]
    #chexpert = iid2chexpert[id]
    split = iid2split[id]
    return [binary1,binary2, captions, id, split]


def make_arrow_iu_xray(data, dataset_name, data_root, save_dir):
    print(f"+ Pre-processing {dataset_name}...")
    iid2captions = defaultdict(list)
    iid2imgs = defaultdict(list)
    iid2chexpert = defaultdict(list)
    iid2split = dict()
    with open(data_root + '/labels_14.pickle','rb') as f:
        labels = pickle.load(f)

    for split, split_data in data.items():
        for sample in split_data:
            iid2captions[sample["id"]].extend([sample["report"]])
            array = sample["id"][:-2].split('-')
            modified_id = array[0] + '-' + array[1]
            #iid2chexpert[sample["id"]].extend(labels[modified_id])
            iid2imgs[sample["id"]].extend(sample["image_path"])
            iid2split[sample["id"]] = split

    num_samples = len(iid2captions)
    img_paths = [path for v in tqdm(iid2imgs.values()) for path in v if os.path.exists(os.path.join(data_root,'images',path))]
    print(f"+ {len(img_paths)} images / {num_samples} annotations")
    bs = [path2rest_iu_xray(id, iid2captions, iid2imgs, iid2chexpert, iid2split, data_root) for id in tqdm(iid2captions)]
    print('clean before:', len(bs))
    bs = [sample for sample in bs if sample is not None]
    print('clean after', len(bs))

    for split in ["train", "val", "test"]:
    #for split in ["train"]:
        batches = [b for b in bs if b[-1] == split]
        print(f'The number of images in {split} set:',len(batches))
        #dataframe = pd.DataFrame(batches, columns=["image", "caption", "image_id", "chexpert", "split"])
        table = pd.DataFrame(batches, columns=["image1", "image2", "caption", "image_id", "split"])
        table = pa.Table.from_pandas(table)

        os.makedirs(save_dir, exist_ok=True)
        save_file = f"{save_dir}/{dataset_name}_{split}.arrow"
        print(f'Writing {save_file}...')
        with pa.OSFile(save_file, "wb") as sink:
            with pa.RecordBatchFileWriter(sink, table.schema) as writer:
                writer.write_table(table)


if __name__ == '__main__':
    data_root = sys.argv[1] + '/iu_xray'
    save_dir = sys.argv[2]
    ann_path = os.path.join(data_root, 'annotation.json')
    ann = json.loads(open(ann_path, 'r').read())
    make_arrow_iu_xray(ann,'iu_xray', data_root, save_dir)
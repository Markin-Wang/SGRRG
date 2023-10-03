from .base_dataset import BaseDatasetArrow, BaseDataset
from PIL import Image
import numpy as np

class MIMICMultiImageDatasetArrow(BaseDatasetArrow):
    def __init__(self, *args, split="", **kwargs):
        assert split in ["train", "val", "test"]
        self.split = split

        if split == "train":
            name = "mimic_cxr_train_filter"
        elif split == "val":
            name = "mimic_cxr_val"
        elif split == "test":
            name = "mimic_cxr_test"
        else:
            name = None

        super().__init__(*args, **kwargs, split=split, name=name, text_column_name="caption")

    def __getitem__(self, index):
        suite = self.get_suite(index)

        if "test" in self.split or 'val' in self.split:
            iid = self.table["image_id"][index].as_py()
            # iid = int(iid.split(".")[0].split("_")[-1])
            suite.update({"iid": iid})

        return suite


class MimiccxrSingleImageDataset(BaseDataset):
    def __getitem__(self, idx):
        example = self.examples[idx]
        image_id = example['id']
        image_path = example['image_path']
        label = np.array(self.labels[image_id]).astype(np.float32)
        try:
            image = Image.open(os.path.join(self.image_dir, image_path[0])).convert('RGB')
        # image = Image.open(os.path.join(self.image_dir, image_path[0])).convert('RGB')
        # image = cv2.cvtColor(cv2.imread(os.path.join(self.image_dir, image_path[0])), cv2.COLOR_BGR2RGB)
        # image = Image.fromarray(image)
        except IOError:
            print('image path', image_path[0])

        if self.transform is not None:
            image = self.transform(image)
        report_ids = example['ids']
        report_masks = example['mask']
        seq_length = len(report_ids)
        sample = (image_id, image, report_ids, report_masks, seq_length, label)
        return sample


class MimiccxrSingleImageClsDataset(BaseDataset):
    def __init__(self, args, split, transform=None, vis=False):
        self.image_dir = os.path.join(args.data_dir, args.dataset_name, 'images')
        self.ann_path = os.path.join(args.data_dir, args.dataset_name, 'annotation.json')
        self.split = split
        self.ann = json.loads(open(self.ann_path, 'r').read())
        self.examples = self.ann[self.split]
        self.labels_path = os.path.join(args.data_dir, args.dataset_name, args.label_path)
        self.labels = json.loads(open(self.labels_path, 'r').read())
        self.transform = transform
        self._labels = [self.labels[e['id']] for e in self.examples]
        self.vis = vis

    def __getitem__(self, idx):
        example = self.examples[idx]
        image_id = example['id']
        label = np.array(self.labels[image_id]).astype(np.float32)
        image_path = example['image_path']
        image = Image.open(os.path.join(self.image_dir, image_path[0])).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        # report_ids = example['ids']
        # report_masks = example['mask']
        # seq_length = len(report_ids)
        # sample = (image_id, image, report_ids, report_masks, seq_length)
        if self.vis:
            return image, image_path, label
        else:
            return image, label

    def __len__(self):
        return len(self.examples)

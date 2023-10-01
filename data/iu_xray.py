from .base_dataset import BaseDatasetArrow, BaseDataset
from PIL import Image
import numpy as np

class IuxrayMultiImageDatasetArrow(BaseDatasetArrow):
    def __init__(self, *args, split="", **kwargs):
        assert split in ["train", "val", "test"]
        self.split = split

        if split == "train":
            name = "iu_xray_train"
        elif split == "val":
            name = "iu_xray_val"
        elif split == "test":
            name = "iu_xray_test"
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


class IuxrayMultiImageDataset(BaseDataset):
    def __getitem__(self, idx):
        example = self.examples[idx]
        image_id = example['id']
        image_path = example['image_path']
        array = image_id.split('-')
        modified_id = array[0] + '-' + array[1]
        label = np.array(self.labels[modified_id]).astype(np.float32)
        # image_path = example['image_path']
        # image_1 = Image.open(os.path.join(self.image_dir, image_path[0])).convert('RGB')
        # image_2 = Image.open(os.path.join(self.image_dir, image_path[1])).convert('RGB')
        if 'images' in example:
            image_1, image_2 = example['images'][0], example['images'][1]
        else:
            image_1 = Image.open(os.path.join(self.image_dir, image_path[0])).convert('RGB')
            image_2 = Image.open(os.path.join(self.image_dir, image_path[1])).convert('RGB')
            self.examples[idx]['images'] = [image_1, image_2]
        if self.transform is not None:
            image_1 = self.transform(image_1)
            image_2 = self.transform(image_2)
        image = torch.stack((image_1, image_2), 0)
        report_ids = example['ids']
        report_masks = example['mask']
        seq_length = len(report_ids)
        sample = (image_id, image, report_ids, report_masks, seq_length, label)
        return sample
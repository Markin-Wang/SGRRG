from .base_dataset import BaseDatasetArrow
from PIL import Image
import numpy as np

class MIMICMultiImageDatasetArrow(BaseDatasetArrow):
    def __init__(self, *args, split="", **kwargs):
        assert split in ["train", "val", "test"]
        self.split = split

        if split == "train":
            name = "mimic_cxr_train_filter_att"
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
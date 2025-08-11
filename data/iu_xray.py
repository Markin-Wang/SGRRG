from .base_dataset import BaseDatasetArrow
from PIL import Image
import numpy as np

class IuxrayMultiImageDatasetArrow(BaseDatasetArrow):
    def __init__(self, *args, split="", **kwargs):
        assert split in ["train", "val", "test", "all"]
        self.split = split

        if split == "train":
            name = "iu_xray_all"
        elif split == "val":
            name = "iu_xray_all"
        elif split == "test":
            name = "iu_xray_all"
        elif split == "all":
            name = ["iu_xray_train", "iu_xray_val", "iu_xray_test"]
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


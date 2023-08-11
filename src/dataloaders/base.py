from abc import *
from collections import namedtuple
from torch.utils.data import DataLoader


class BaseLoader(metaclass=ABCMeta):
    def __init__(self):
        super().__init__()

    @classmethod
    def get_name(cls):
        return cls.name

    def get_dataloader(self)-> DataLoader:
        pass

    @ property
    def categorical_ids(self):
        return self.dataset.categorical_ids

    @ property
    def standardize_dict(self):
        return self.dataset.standardize_dict 
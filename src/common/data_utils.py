import torch.utils.data as data_utils

class CustomRandomSampler(data_utils.Sampler):
    def __init__(self, n, rng):
        super().__init__(data_source=[]) # dummy
        self.n = n
        self.rng = rng

    def __len__(self):
        return self.n

    def __iter__(self):
        indices = list(range(self.n))
        self.rng.shuffle(indices)
        return iter(indices)

    def get_rng_state(self):
        return self.rng.getstate()

    def set_rng_state(self, state):
        return self.rng.setstate(state)

class CustomIndexSampler(data_utils.Sampler):
    def __init__(self, indices, rng, shuffle):
        super().__init__(data_source=[]) # dummy
        self.indices = indices
        self.rng = rng
        self.shuffle = shuffle

    def __len__(self):
        return len(self.indices)

    def __iter__(self):
        indices = self.indices
        if self.shuffle:
            self.rng.shuffle(indices)
        return iter(indices)

    def get_rng_state(self):
        return self.rng.getstate()

    def set_rng_state(self, state):
        return self.rng.setstate(state)
import numpy as np


class AugmentDataLoader:
    def __init__(self, batch_list, shuffle=True):
        self.batch_list = batch_list
        self.shuffle = shuffle

        self._nun_samples = len(self.batch_list)
        self._cnt = 0
        self._id_list = np.arange(0, self._nun_samples)

    def __len__(self):
        return self._nun_samples

    def __iter__(self):
        yield self.__next__()

    def __next__(self):
        self._cnt = (self._cnt + 1) % self._nun_samples
        if self._cnt == 0 and self.shuffle:
            np.random.shuffle(self._id_list)
        idx = self._id_list[self._cnt]
        return self.batch_list[idx]

from src.data import Data


BATCH_SIZE = 1  # for full loader


class Loader:
    """
    TODO: torch geomtric data structure.
    """
    def __init__(self, batch_size: int, data: Data, **kwargs):
        self.batch_size = batch_size
        self.data = data

    def __iter__(self, *args, **kwargs):
        raise NotImplementedError


class FullLoader(Loader):
    """
    r"Load entire graph each time."
    """
    def __init__(self, data: Data):
        super().__init__(batch_size=BATCH_SIZE, data=data)

    def __iter__(self):
        return self.data


# class BatchLoader(Loader):
#     """
#     r"Load entire graph each time."
#     """
#     def __init__(self, data: Data):
#         super().__init__(batch_size=BATCH_SIZE, data=data)
#         self.x = self.data.x
#         self.adj = self.data.adj
#         self.label = self.data.y
#
#     def __iter__(self, batch):
#         self.x = self.data.x[batch]
#         self.adj = self.data.adj[batch, :][:, batch]
#         self.label = self.data.label[batch]
#         return self.data

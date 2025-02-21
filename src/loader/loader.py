BATCH_SIZE = 1  # for full loader


class Loader:
    """
    TODO: torch geomtric data structure.
    """
    def __init__(self, batch_size: int, data, **kwargs):
        self.batch_size = batch_size
        self.data = data

    def __iter__(self, *args, **kwargs):
        raise NotImplementedError


class FullLoader(Loader):
    """
    r"Load entire graph each time."
    """
    def __init__(self, data):
        super().__init__(batch_size=BATCH_SIZE, data=data)

    def __iter__(self):
        return self.data

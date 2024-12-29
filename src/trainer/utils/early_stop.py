class EarlyStopper:
    r""" Early stopping. This class has two status 'stop' and 'save'. Use the update function to update the status.
        stop: Whether stop the training. When the maximum patience is reached, the stop=True; else, stop=False.
        save: Whether save the model. When the current_value is better than the historical best value, then save=True.

    Args:
        patience (int): The maximum tolerance.
            (default: 20)
        type (str): Stop at the maximum ('max') or minimum ('min') value.
            (default: 'max')
    """
    def __init__(self, patience: int = 20, type: str = "min") -> None:
        self.patience = patience
        assert type in {'max', 'min'}
        self.type = type

        self.stop = False
        self.save = False

        self._best_value = None
        self._n_wait = 0

    def update(self, current_value) -> None:
        r"""Update the status `save` and `stop`."""
        if self._best_value is None:  # the first call of the method.
            self._best_value = current_value
            self._n_wait = 0
            return

        if self.type == "min" and current_value < self._best_value \
                or self.type == "max" and current_value > self._best_value:
            self._n_wait = 0
            self._best_value = current_value
            self.save = True
        else:
            self._n_wait += 1
            self.save = False

        if self._n_wait == self.patience:
            print('Early stopped!')
            self.stop = True

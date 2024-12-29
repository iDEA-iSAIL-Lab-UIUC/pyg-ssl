class BaseEvaluator:
    r"""Base class for Evaluator."""
    def __call__(self, *args, **kwargs):
        raise NotImplementedError

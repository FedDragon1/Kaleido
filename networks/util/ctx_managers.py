class ChainError:
    """
    Raises the error passed in from the error raised
    inside the context manager.
    """

    def __init__(self, exception):
        self.exception = exception

    def __enter__(self):
        ...

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            raise self.exception from exc_val

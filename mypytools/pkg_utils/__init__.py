class NullClass:
    """A dummy class to avoid hard ImportError"""

    def __init__(self, name: str, info: str):
        self.name = name
        self.info = info

    def __call__(self, *args, **kwargs):
        raise ImportError(f"{self.name} is not available: {self.info}")

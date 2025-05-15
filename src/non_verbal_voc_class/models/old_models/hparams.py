class HParams:
    """
    This class is used to transform a dictionary into a HParams object with attributes.
    """
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
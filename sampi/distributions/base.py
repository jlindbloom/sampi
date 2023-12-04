



class RV:
    """Represents a generic random variable.
    """
    def __init__(self, name, size):

        self.name = name



class ContinuousRV(RV):
    """Represents a general multivariate continuous random variable.
    """
    def __init__(self, name, size):

        super().__init__(name)

    def logpdf(self, x):
        """Evaluates the log pdf (up to a normalization constant) at the point x.
        """
        raise NotImplementedError

    def normalized_logpdf(self, x):
        """Evaluates the log pdf (including up to normalization constant) at the point
        """
        raise NotImplementedError


















"""Preprocessing Module
"""

class MissingTransformer():
    """Process the columns with missing values
    """
    def __init__(self, fill_value='median'):
        self.fill_value = fill_value

    def fit(self, x):
        pass

    def transform(self, x):
        pass

    def fit_transform(self, x):
        pass


class SmartTransformer():
    """Transform the data automatically based on best practices
    """
    def __init__(self):
        self.trained = False

    def fit(self, x):
        pass

    def transform(self, x):
        pass

    def fit_transform(self, x):
        pass

"""Module with custom exceptions"""


class GitIsDirtyException(Exception):
    """Exception raised when there are uncommited changes in current repo"""

    def __init__(self, message="There are uncommited changes in repo!"):
        super().__init__(message)


class BadPickleLoaderDataType(Exception):
    """Exception raised when initializing PickleLoader with inadequate `data_type` argument."""

    def __init__(self, arg_value):
        message = f"Bad 'data_type' argument value: {arg_value}. Should be one of: 'data', 'meta'."
        super().__init__(message)


class TfIdfNotFitted(Exception):
    """Exception raised when trying to transform with not fitted model"""

    def __init__(self, message="This model has not been fitted yet. Call `fit` first."):
        super().__init__(message)

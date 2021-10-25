class GitIsDirtyException(Exception):
    """Exception raised when there are uncommited changes in current repo"""

    def __init__(self, message="There are uncommited changes in repo!"):
        super().__init__(message)

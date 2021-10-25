from git import Repo
from collections import namedtuple


class GitParser:
    """Class providing interface for reading git info and status"""

    def __init__(self, repo_path='.'):
        self._repo = Repo(repo_path)
        self._head = self._repo.head
        self._master = self._head.reference

        self.working_dir = self._master.repo.working_dir
        self.branch_name = self._master.name
        self.commit_message = self._master.commit.message
        self.commit_sha = self._master.commit.hexsha

    def get_info(self):
        GitDetails = namedtuple("git_details", "dir branch message sha")
        return GitDetails(self.working_dir, self.branch_name, self.commit_message, self.commit_sha)

    def is_dirty(self):
        return self._repo.is_dirty()

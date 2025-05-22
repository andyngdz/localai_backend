"""Custom TQDM"""

from tqdm import tqdm


class DownloadTPDM(tqdm):
    """
    A custom progress bar class extending `tqdm` for download operations.
    """

    def __init__(self, *args, **kwargs):
        # Disable ascii to keep it clean, or you can enable it
        kwargs.setdefault("ascii", True)
        kwargs.setdefault("mininterval", 0.5)  # update at most twice per second
        super().__init__(*args, **kwargs)

    def update(self, n=1):
        super().update(n)
        # Optionally: print extra info per update if you want
        # print(f"Downloaded {self.n}/{self.total} bytes")

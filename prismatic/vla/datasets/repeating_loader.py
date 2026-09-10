"""
repeating_loader.py

Wraps a DataLoader to cycle through data indefinitely, matching the behavior of
RLDS IterableDatasets which loop forever.
"""


class InfiniteDataLoader:
    """Wraps a finite DataLoader to cycle through data indefinitely."""

    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.iterator = iter(dataloader)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            return next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.dataloader)
            return next(self.iterator)

    def __len__(self):
        return len(self.dataloader)

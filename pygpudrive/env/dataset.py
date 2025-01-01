import os
import random
from typing import List, Iterator
from dataclasses import dataclass


@dataclass
class SceneDataLoader:
    root: str
    batch_size: int
    dataset_size: int
    sample_with_replacement: bool = False
    file_prefix: str = "tfrecord"
    seed: int = 42

    """
    A data loader for sampling batches of traffic scnearios from a directory of files.

    Attributes:
        root (str): Path to the directory containing scene files.
        batch_size (int): Number of scenes per batch (should be equal to number of worlds in the env).
        dataset_size (int): Maximum number of files to include in the dataset.
        sample_with_replacement (bool): Whether to sample files with replacement.
        file_prefix (str): Prefix for scene files to include in the dataset.
        seed (int): Seed for random number generator to ensure reproducibility.
    """

    def __post_init__(self):
        # Validate the path
        if not os.path.exists(self.root):
            raise FileNotFoundError(
                f"The specified path does not exist: {self.root}"
            )

        # Set the random seed for reproducibility
        random.seed(self.seed)

        # Create the dataset from valid files in the directory
        self.dataset = [
            os.path.join(self.root, scene)
            for scene in sorted(os.listdir(self.root))
            if scene.startswith(self.file_prefix)
        ]

        # Adjust dataset size based on the provided dataset_size
        self.dataset = self.dataset[
            : min(self.dataset_size, len(self.dataset))
        ]

        # Initialize state for iteration
        self._reset_indices()

    def _reset_indices(self):
        """Reset indices for sampling."""
        if self.sample_with_replacement:
            self.indices = None  # No need to track indices for replacement
        else:
            self.indices = list(range(len(self.dataset)))
            random.shuffle(self.indices)
        self.current_index = 0

    def __iter__(self) -> Iterator[List[str]]:
        self._reset_indices()
        return self

    def __len__(self):
        """Get the number of batches in the dataloader."""
        return len(self.dataset) // self.batch_size

    def __next__(self) -> List[str]:
        if self.sample_with_replacement:
            batch = random.choices(self.dataset, k=self.batch_size)
        else:
            if self.current_index >= len(self.indices):
                raise StopIteration

            # Get the next batch of indices
            end_index = min(
                self.current_index + self.batch_size, len(self.indices)
            )
            batch_indices = self.indices[self.current_index : end_index]
            self.current_index = end_index

            # Retrieve the corresponding scenes
            batch = [self.dataset[i] for i in batch_indices]

        return batch


# Example usage
if __name__ == "__main__":
    data_loader = SceneDataLoader(
        root="data/processed/training",
        batch_size=100,
        dataset_size=1000,
        sample_with_replacement=False,
    )

    for batch in data_loader:
        print(batch)

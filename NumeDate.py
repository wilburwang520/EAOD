import numpy as np

class AE_Data:
    def __init__(self, X):
        self.batch_start_index = 0
        self.X = X
        self.data_size = len(X)
        self.epochs_completed = 0

    def next_batch(self, batch_size):
        end_index = self.batch_start_index + batch_size
        if end_index > self.data_size:
            # Shuffle the data
            perm = np.arange(self.data_size)
            np.random.shuffle(perm)
            self.X = self.X[perm]

            self.epochs_completed += 1
            self.batch_start_index = 0

        start_index = self.batch_start_index
        batch_X = self.X[start_index: end_index]

        return batch_X
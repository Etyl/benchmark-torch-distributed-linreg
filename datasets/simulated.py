from benchopt import BaseDataset
import numpy as np
import torch.nn as nn

from benchmark_utils.dataset_utils import get_dataloader



class Dataset(BaseDataset):
    name = "simulated"

    parameters = {
        'n': [16*1024],
        'd1': [400],
    }
    requirements = ["numpy"]

    def get_data(self):
        rng = np.random.RandomState(42)

        X = rng.randn(self.n, self.d1)
        W_linear = rng.randn(self.d1, self.d1)
        Y = X @ W_linear

        dataloader = get_dataloader(X, Y, batch_size=self.n)
        self.model = nn.Linear(
            self.dataloader.dataset.X.shape[1],
            self.dataloader.dataset.Y.shape[1],
            bias=False,
        )

        return dict(dataloader=dataloader, model=self.model)

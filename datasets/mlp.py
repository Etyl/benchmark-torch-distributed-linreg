from benchopt import BaseDataset
import numpy as np
import torch.nn as nn

from benchmark_utils.dataset_utils import TorchDataset


class Dataset(BaseDataset):
    name = "simulated"

    parameters = {
        'n': [16*1024],
        'd': [400],
        'layers': [1],
        'bias': [False],
    }
    requirements = ["numpy"]

    def get_data(self):
        rng = np.random.RandomState(42)

        X = rng.randn(self.n, self.d)
        W_linear = rng.randn(self.d, self.d)
        Y = X @ W_linear

        dataset = TorchDataset(X, Y)

        layers = []
        for _ in range(self.layers):
            layers.append(nn.Linear(self.d, self.d, bias=self.bias))
            layers.append(nn.ReLU())
        model = nn.Sequential(*layers)

        return dict(dataset=dataset, model=model)

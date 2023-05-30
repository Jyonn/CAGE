# C5 Module

## Usage

```python
import torch
from torch import nn
from c5rec import C5


class BaseRecommenderConfig:
    def __init__(
            self,
            dim=64,  # embedding dimension
            ways=None,  # vocabulary sizes per layer
            alpha=1.0,  # weighted add
            beta=1.0,  # commitment cost
            omega=1.0,  # quantization loss weight
    ):
        self.dim = dim
        self.ways = ways
        self.alpha = alpha
        self.beta = beta
        self.omega = omega


class BaseRecommender(nn.Module):
    def __init__(self, config):
        super(BaseRecommender, self).__init__()
        self.config = config

        # first line: C5 initialization
        self.c5 = C5(
            dim=config.dim,
            ways=config.ways,
            alpha=config.alpha,
            beta=config.beta,
        )

        self.enc = ...

    def forward(self, items, users):
        item_embs = self.get_item_embs(items)
        user_embs = self.get_user_embs(users)

        # second line: embedding quantization
        item_embs, qloss = self.c5(item_embs)

        out = self.enc(item_embs, user_embs)
        loss = self.pred(out)

        # third line: update loss
        return loss + qloss * self.conf.omega

    def pred(self, out) -> torch.Tensor:
        ...

    def get_item_embs(self, items) -> torch.Tensor:
        ...

    def get_user_embs(self, users) -> torch.Tensor:
        ...
```
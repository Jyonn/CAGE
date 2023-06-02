# CADRE Module

## Usage

```python
import torch
from torch import nn
from cadre import Cadre


class BaseRecommenderConfig:
    def __init__(
            self,
            dim=64,  # embedding dimension
            entries=None,  # code entries per layer
            alpha=1.0,  # weighted add
            beta=1.0,  # commitment cost
            omega=1.0,  # quantization loss weight
    ):
        self.dim = dim
        self.entries = entries
        self.alpha = alpha
        self.beta = beta
        self.omega = omega


class BaseRecommender(nn.Module):
    def __init__(self, config):
        super(BaseRecommender, self).__init__()
        self.config = config

        # first line: module initialization
        self.cadre = Cadre(
            dim=config.dim,
            entries=config.entries,
            alpha=config.alpha,
            beta=config.beta,
        )

        self.enc = ...

    def forward(self, items, users):
        item_embs = self.get_item_embs(items)
        user_embs = self.get_user_embs(users)

        # second line: embedding quantization
        item_embs, qloss = self.cadre(item_embs)

        out = self.enc(item_embs, user_embs)
        loss = self.pred(out)

        # third line: loss updating
        return loss + qloss * self.conf.omega

    def pred(self, out) -> torch.Tensor:
        ...

    def get_item_embs(self, items) -> torch.Tensor:
        ...

    def get_user_embs(self, users) -> torch.Tensor:
        ...
```
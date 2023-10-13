# Cage Module

## Usage

```python
import torch
from torch import nn
from cage import Cage


class CageConfig:
    def __init__(
            self,
            dim=64,  # embedding dimension
            entries=None,  # code entries per layer
            alpha=1.0,  # weighted add
            beta=1.0,  # commitment cost
    ):
        self.dim = dim
        self.entries = entries
        self.alpha = alpha
        self.beta = beta


class BaseRecommender(nn.Module):
    def __init__(self, user_config, item_config, omega=1.0):
        super(BaseRecommender, self).__init__()

        # cage initialization
        self.item_cage = Cage(
            dim=item_config.dim,
            entries=item_config.entries,
            alpha=item_config.alpha,
            beta=item_config.beta,
        )

        self.user_cage = Cage(
            dim=user_config.dim,
            entries=user_config.entries,
            alpha=user_config.alpha,
            beta=user_config.beta,
        )

        self.omega = omega

        self.enc = ...

    def forward(self, items, users):
        item_embs = self.get_item_embs(items)
        user_embs = self.get_user_embs(users)

        # second line: embedding quantization
        item_embs, item_loss = self.item_cage(item_embs)
        user_embs, user_loss = self.user_cage(user_embs)

        out = self.enc(item_embs, user_embs)
        loss = self.pred(out)

        # third line: loss updating
        return loss + (item_loss + user_loss) * self.omega

    def pred(self, out) -> torch.Tensor:
        ...

    def get_item_embs(self, items) -> torch.Tensor:
        ...

    def get_user_embs(self, users) -> torch.Tensor:
        ...
```
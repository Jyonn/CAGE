import warnings

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from common import TransformLayer, DecoderLayer, C5Quantization, C5Classification, C5Module


class C5(C5Module):
    """
    Cascade Clusterer
    """
    def __init__(
            self,
            embed_dim,
            vocab_size,
            num_layers=None,
            cluster_sizes=None,  # ex. [100, 10]
            weighted_add=1,  # alpha
            commitment_cost=0.25,  # beta
            layer_connect=False,
            layer_loss=True,
    ):
        super().__init__()

        if cluster_sizes is not None and not isinstance(cluster_sizes, list):
            cluster_sizes = str(cluster_sizes)
            cluster_sizes = [int(x) for x in cluster_sizes.split('-')]

        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.num_layers = num_layers  # type: int
        self.cluster_sizes = cluster_sizes  # type: list[int]
        self.weighted_add = weighted_add
        self.commitment_cost = commitment_cost
        self.layer_connect = layer_connect
        self.layer_loss = layer_loss

        assert vocab_size >= 1, "vocab_size must be greater than 1"
        assert cluster_sizes is not None or num_layers is not None, "num_layers or cluster_sizes must be specified"

        self.set_cluster_size()

        # construct codebooks
        self.codebooks = nn.ModuleList()  # ex. [nn.Embedding(4, D), nn.Embedding(2, D)]
        for i in range(self.num_layers):
            self.codebooks.append(nn.Embedding(self.cluster_sizes[i], embed_dim))

        # layers for classification
        self.transform_layer = TransformLayer(
            embed_dim=self.embed_dim,
            activation_function='relu',
        )
        self.decoder_layer = DecoderLayer(
            embed_dim=self.embed_dim,
            vocab_size=self.vocab_size,
        )
        # decoder layers for each layer
        self.codebook_decoders = nn.ModuleList()  # ex. [DecoderLayer(D, 4), DecoderLayer(D, 2)]
        for i in range(self.num_layers):
            self.codebook_decoders.append(DecoderLayer(
                embed_dim=self.embed_dim,
                vocab_size=self.cluster_sizes[i],
            ))

    def set_cluster_size(self):
        if self.cluster_sizes is not None:
            self.num_layers = len(self.cluster_sizes)
            return

        assert self.num_layers >= 0 and isinstance(self.num_layers, int), "num_layers must be a non-negative integer"

        # calculate cluster sizes
        if 2 ** self.num_layers > self.vocab_size:
            warnings.warn(f"num_layers is too large, set to {self.num_layers}")
            self.num_layers = int(np.log2(self.vocab_size))

        top_cluster_size = int(self.vocab_size ** (1.0 / (self.num_layers + 1)) + 0.5)
        self.cluster_sizes = [top_cluster_size]  # ex. [2]
        for i in range(self.num_layers - 1):
            self.cluster_sizes.append(top_cluster_size * self.cluster_sizes[i])
        self.cluster_sizes = self.cluster_sizes[::-1]  # ex. [4, 2]

    def quantize(
            self,
            embeds,
            with_loss=False,
    ) -> C5Quantization:
        compare_embeds = embeds  # for loss calculation

        shape = embeds.shape
        embeds = embeds.view(-1, self.embed_dim)  # [B * ..., D]
        qembeds = []
        qindices = []

        for i in range(self.num_layers):
            dist = torch.cdist(embeds, self.codebooks[i].weight, p=2)
            indices = torch.argmin(dist, dim=-1).unsqueeze(1)
            placeholder = torch.zeros(indices.shape[0], self.cluster_sizes[i], device=embeds.device)
            placeholder.scatter_(1, indices, 1)
            inner_embeds = torch.matmul(placeholder, self.codebooks[i].weight).view(embeds.shape)
            qembeds.append(inner_embeds.view(shape))
            qindices.append(indices.view(shape[:-1]))
            if self.layer_connect:
                embeds = inner_embeds

        output = C5Quantization(qembeds, indices=qindices)
        if output.mean != 0:
            output.mean += embeds * self.weighted_add

        if not with_loss:
            return output

        q_loss = torch.tensor(0, dtype=torch.float, device=embeds.device)
        for i in range(self.num_layers):
            q_loss += F.mse_loss(qembeds[i].detach(), compare_embeds) * self.commitment_cost \
                      + F.mse_loss(qembeds[i], compare_embeds.detach())
            if self.layer_connect:
                compare_embeds = qembeds[i]
        output.loss = q_loss

        return output

    def classify(
            self,
            embeds,
            indices=None,
    ) -> C5Classification:
        embeds = self.transform_layer(embeds)
        scores = self.decoder_layer(embeds)

        cls_loss = torch.tensor(0, dtype=torch.float, device=embeds.device)
        if indices:
            for i in range(self.num_layers):
                layer_scores = self.codebook_decoders[i](embeds)
                if self.layer_loss:
                    cls_loss += F.cross_entropy(layer_scores, indices[i].view(-1), reduction='mean')

        return C5Classification(scores, layer_loss=cls_loss)

    def __call__(self, *args, **kwargs):
        return self.quantize(*args, **kwargs)

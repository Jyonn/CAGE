import torch
from torch import nn
from transformers.activations import ACT2FN


class C5Quantization:
    def __init__(self, embeds, loss=None, indices=None):
        self.embeds = embeds
        self.loss = loss
        self.indices = indices
        if len(self.embeds) > 0:
            if isinstance(self.embeds, torch.Tensor):
                self.mean = torch.mean(self.embeds, dim=2)
            else:
                self.mean = torch.mean(torch.stack(self.embeds), dim=0)
        else:
            self.mean = 0


class C5Classification:
    def __init__(self, scores, layer_loss=None, indices=None):
        self.scores = scores
        self.indices = indices
        self.layer_loss = layer_loss


class C5Module(nn.Module):
    def quantize(
            self,
            embeds,
            with_loss=False,
    ) -> C5Quantization:
        raise NotImplementedError

    def classify(
            self,
            embeds,
            indices=None,
    ) -> C5Classification:
        raise NotImplementedError


class TransformLayer(nn.Module):
    """
    Transform layer for Classifier
    """

    def __init__(
            self,
            embed_dim,
            activation_function,
            layer_norm_eps=None,
    ):
        super(TransformLayer, self).__init__()
        self.transform = nn.Linear(embed_dim, embed_dim)
        self.transform_act_fn = ACT2FN[activation_function]
        self.layer_norm = nn.LayerNorm(embed_dim, eps=layer_norm_eps or 1e-5)

    def forward(self, hidden_states) -> torch.Tensor:
        hidden_states = self.transform(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        return hidden_states


class DecoderLayer(nn.Module):
    """
    Decoder layer for Classifier, projecting hidden states to vocab_size
    """

    def __init__(
            self,
            embed_dim,
            vocab_size,
    ):
        super(DecoderLayer, self).__init__()
        self.decoder = nn.Linear(embed_dim, vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(vocab_size), requires_grad=True)
        self.decoder.bias = self.bias

    def forward(self, hidden_states) -> torch.Tensor:
        return self.decoder(hidden_states)

    def set_values(self, decoder_weights: torch.Tensor, decoder_bias: torch.Tensor):
        self.decoder.weight.data = decoder_weights.data
        self.bias.data = decoder_bias.data
        self.decoder.bias = self.bias

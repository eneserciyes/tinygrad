from typing import Tuple, Dict
from dataclasses import dataclass

from tinygrad import Tensor, nn
from nets import Encoder, Decoder

Batch = Dict[str, Tensor]

class Loss:
  def __init__(self, **kwargs):
    self.total_loss = sum(kwargs.values())
    self.intermediate_losses = {k: v.item() for k, v in kwargs.items()}

  def __truediv__(self, value):
    for k, v in self.intermediate_losses.items():
      self.intermediate_losses[k] = v / value
    self.total_loss = self.total_loss / value
    return self

@dataclass
class TokenizerEncoderOutput:
    z: Tensor
    z_quantized: Tensor
    tokens: Tensor


class Tokenizer:
  def __init__( self, vocab_size: int, embed_dim: int, encoder: Encoder, decoder: Decoder, with_lpips=False,) -> None:
    self.vocab_size = vocab_size
    self.encoder = encoder
    self.pre_quant_conv = nn.Conv2d(encoder.config.z_channels, embed_dim, 1)
    self.embedding = nn.Embedding(vocab_size, embed_dim)
    self.post_quant_conv = nn.Conv2d(embed_dim, decoder.config.z_channels, 1)
    self.decoder = decoder
    self.embedding.weight = Tensor.uniform((vocab_size, embed_dim), -1.0 / vocab_size, 1.0 / vocab_size)
    # TODO: implement LPIPS
    self.lpips = lambda x, y: Tensor(0.0)

  def compute_loss(self, batch: Batch) -> Loss:
    assert self.lpips is not None, "LPIPS is not implemented"

    obs = batch["observations"]
    obs = obs.reshape(obs.shape[0] * obs.shape[1], *obs.shape[2:]).mul(2).sub(1)
    z, z_q, reconstructions = self(obs, should_preprocess=False)

    beta = 1.0
    commit_loss = (z.detach() - z_q).pow(2).mean() + beta * (z - z_q.detach()).pow(2).mean()
    reconstruction_loss = (reconstructions - obs).abs().mean()
    perceptual_loss = self.lpips(reconstructions, obs).mean()

    return Loss(commit_loss=commit_loss, reconstruction_loss=reconstruction_loss, perceptual_loss=perceptual_loss)

  def __call__( self, x: Tensor, should_preprocess: bool = False, should_postprocess: bool = False,) -> Tuple[Tensor, Tensor, Tensor]:
    outputs = self.encode(x, should_preprocess)
    dec_input = outputs.z + (outputs.z_quantized - outputs.z).detach()
    reconstruction = self.decode(dec_input, should_postprocess)
    return outputs.z, outputs.z_quantized, reconstruction

  def encode(self, x: Tensor, should_preprocess: bool = False) -> TokenizerEncoderOutput:
    if should_preprocess: x = x.mul(2).sub(1)
    shape = x.shape
    x = x.reshape(-1, *shape[-3:])
    z = self.encoder(x)
    z = self.pre_quant_conv(z)
    b, e, h, w = z.shape
    z_flattened = z.permute(0, 2, 3, 1).reshape(-1, e)
    dist_to_embeddings = (
        z_flattened.pow(2).sum(1, keepdim=True) + self.embedding.weight.pow(2).sum(1) - 2 * z_flattened.matmul(self.embedding.weight.transpose())
    )
    tokens = dist_to_embeddings.argmin(-1, keepdim=True)
    z_q = (self.embedding(tokens).reshape(b, h, w, e).permute(0, 3, 1, 2).contiguous())
    z = z.reshape(*shape[:-3], *z.shape[1:])
    z_q = z_q.reshape(*shape[:-3], *z_q.shape[1:])
    tokens = tokens.reshape(*shape[:-3], -1)
    return TokenizerEncoderOutput(z, z_q, tokens)

  def decode(self, z_q: Tensor, should_postprocess: bool = False) -> Tensor:
    shape = z_q.shape
    z_q = z_q.reshape(-1, *shape[-3:])
    z_q = self.post_quant_conv(z_q)
    rec = self.decoder(z_q)
    rec = rec.reshape(*shape[:-3], *rec.shape[1:])
    if should_postprocess: rec = rec.add(1).div(2)
    return rec

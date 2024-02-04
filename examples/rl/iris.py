from tinygrad import nn, Tensor, TinyJit
from extra.models.resnet import BasicBlock as ResBlock
from typing import Tuple, Dict, List
from types import SimpleNamespace
from dataclasses import dataclass

import tinygrad

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
class EncoderDecoderConfig:
  resolution: int
  in_channels: int
  z_channels: int
  ch: int
  ch_mult: List[int]
  num_res_blocks: int
  attn_resolutions: List[int]
  out_ch: int
  dropout: float


class Encoder:
  def __init__(self, cfg: EncoderDecoderConfig) -> None:
    self.cfg = cfg
    self.num_resolutions = len(cfg.ch_mult)
    self.conv_in = nn.Conv2d(cfg.in_channels, cfg.ch, 3, 1, 1)

    curr_res = cfg.resolution
    in_ch_mult = (1,) + tuple(cfg.ch_mult)
    self.down = []
    for i in range(self.num_resolutions):
      block, attn = [], []
      block_in, block_out = cfg.ch * in_ch_mult[i], cfg.ch * in_ch_mult[i + 1]

      for _ in range(cfg.num_res_blocks):
        block.append(ResBlock(block_in, block_out))
        block_in = block_out
        if curr_res in cfg.attn_resolutions:
          attn.append(ResAttnBlock(block_in, 1))

      down = SimpleNamespace()
      if i != self.num_resolutions - 1:
        down.downsample = Downsample(block_in, with_conv=True)  # TODO: implement Downsample
        curr_res //= 2

      down.block = block
      down.attn = attn
      self.down.append(down)

    self.mid = SimpleNamespace()
    self.mid.block_1 = ResBlock(block_in, block_in)
    self.mid.attn_1 = ResAttnBlock(block_in, 1)
    self.mid.block_2 = ResBlock(block_in, block_in)
    self.conv_out = nn.Conv2d(block_in, cfg.z_channels, 3, 1, 1)
    self.norm_out = nn.GroupNorm(32, block_in, eps=1e-6, affine=True)

  def __call__(self, x: Tensor) -> Tensor:
    hs = [self.conv_in(x)]
    for i_level in range(self.num_resolutions):
      for i_block in range(self.cfg.num_res_blocks):
        h = self.down[i_level].block[i_block](hs[-1])
        if len(self.down[i_level].attn) > 0:
          h = self.down[i_level].attn[i_block](h)
        hs.append(h)
        if i_level != self.num_resolutions - 1:
          hs.append(self.down[i_level].downsample(hs[-1]))
    # mid
    h = hs[-1]
    h = self.mid.block_2(self.mid.attn(self.mid.block_1(h)))

    h = self.norm_out(h)
    h = self.conv_out(h)
    return h


class Tokenizer:
  def __init__(
    self,
    vocab_size: int,
    embed_dim: int,
    encoder: Encoder,
    decoder: Decoder,
    with_lpips=False,
  ) -> None:
    self.vocab_size = vocab_size
    self.encoder = encoder
    self.pre_quant_conv = nn.Conv2d(encoder.cfg.z_channels, embed_dim, 1)
    self.embedding = nn.Embedding(vocab_size, embed_dim)
    self.post_quant_conv = nn.Conv2d(embed_dim, decoder.config.z_channels, 1)
    self.decoder = decoder
    self.embedding.weight = Tensor.uniform(
        (vocab_size, embed_dim), -1.0 / vocab_size, 1.0 / vocab_size
    )
    # TODO: implement LPIPS
    # self.lpips = LPIPS().eval() if with_lpips else None

  def forward(
    self,
    x: Tensor,
    should_preprocess: bool = False,
    should_postprocess: bool = False,
  ) -> Tuple[Tensor, Tensor, Tensor]:
    # TODO: check if should_pre/postprocess is needed
    z, z_quantized, _ = self.encode(x, should_preprocess)
    dec_input = z + (z_quantized - z).detach()
    recon = self.decode(dec_input, should_postprocess)
    return z, z_quantized, recon

  def encode(
    self, x: Tensor, should_preprocess: bool = False
  ) -> Tuple[Tensor, Tensor, Tensor]:
    if should_preprocess:
      x = x.mul(2).sub(1)
    z = self.pre_quant_conv(self.encoder(x.reshape(-1, *x.shape[-3:])))
    b, e, h, w = z.shape
    z_flat = z.permute(0, 2, 3, 1).reshape(-1, e)
    dist_to_embeddings = (
      z_flat.pow(2).sum(1, keepdim=True)
      + self.embedding.weight.pow(2).sum(1)
      - 2 * z_flat.matmul(self.embedding.weight.transpose())
    )
    tokens = dist_to_embeddings.argmin(-1, keepdim=True)
    z_q = (
      self.embedding(tokens).reshape(b, h, w, e).permute(0, 3, 1, 2).contiguous()
    )

    z = z.reshape(*x.shape[:-3], *z.shape[1:])
    z_q = z_q.reshape(*x.shape[:-3], *z_q.shape[1:])
    tokens = tokens.reshape(*x.shape[:-3], -1)
    return z, z_q, tokens

  def decode(self, z_q: Tensor, should_postprocess: bool = False) -> Tensor:
    shape = z_q.shape
    z_q = z_q.reshape(-1, *shape[-3:])
    rec = self.decoder(self.post_quant_conv(z_q))
    rec = rec.reshape(*shape[:-3], *rec.shape[1:])
    if should_postprocess:
      rec = rec.add(1).div(2)
    return rec

  def loss(self, batch: Batch) -> Loss:
    obs = batch["observations"]
    obs = obs.reshape(obs.shape[0] * obs.shape[1], *obs.shape[2:]).mul(2).sub(1)
    z, z_q, recon = self.forward(
      obs, should_preprocess=False, should_postprocess=False
    )
    beta = 1.0
    commit_loss = (
      z.detach().sub(z_q).pow(2).mean() + beta * z.sub(z_q.detach()).pow(2).mean()
    )
    recon_loss = (recon - obs).abs().mean()
    # TODO: implement LPIPS
    perceptual_loss = 0.0
    # perceptual_loss = self.lpips(recon, obs).mean()
    return Loss(
      commit_loss=commit_loss,
      recon_loss=recon_loss,
      perceptual_loss=perceptual_loss,
    )

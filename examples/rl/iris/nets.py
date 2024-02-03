from types import SimpleNamespace
from tinygrad import Tensor, nn
from typing import Optional, List
from itertools import chain
from dataclasses import dataclass

def swish(x: Tensor): return x.mul(x.sigmoid())


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
  def __init__(self, config: EncoderDecoderConfig) -> None:
    self.config = config
    self.num_resolutions = len(config.ch_mult)
    temb_ch = 0

    self.conv_in = nn.Conv2d(config.in_channels, config.ch, 3, 1, 1)
    curr_res = config.resolution
    in_ch_mult = (1,) + tuple(config.ch_mult)
    self.down=[]
    for i_level in range(self.num_resolutions):
        block = []
        attn = []
        block_in = config.ch * in_ch_mult[i_level]
        block_out = config.ch * in_ch_mult[i_level + 1]
        for i_block in range(config.num_res_blocks):
            block.append(ResnetBlock(block_in, block_out, dropout=config.dropout, temb_channels=temb_ch))
            block_in = block_out
            if curr_res in config.attn_resolutions:
                attn.append(ResAttnBlock(block_in))
        down = SimpleNamespace()
        down.block=block
        down.attn=attn
        if i_level != self.num_resolutions - 1:
            down.downsample = Downsample(block_in, with_conv=True)
            curr_res //= 2
        self.down.append(down)

    self.mid = SimpleNamespace()
    self.mid.block_1 = ResnetBlock(block_in, block_in, dropout=config.dropout, temb_channels=temb_ch)
    self.mid.attn_1 = ResAttnBlock(block_in)
    self.mid.block_2 = ResnetBlock(block_in, block_in, dropout=config.dropout, temb_channels=temb_ch)

    self.norm_out = nn.GroupNorm(num_groups=32, num_channels=block_in, eps=1e-6, affine=True)
    self.conv_out = nn.Conv2d(block_in, config.z_channels, 3, 1, 1)

  def __call__(self, x: Tensor) -> Tensor:
    temb = None
    hs = [self.conv_in(x)]
    for i_level in range(self.num_resolutions):
        for i_block in range(self.config.num_res_blocks):
            h = self.down[i_level].block[i_block](hs[-1], temb)
            if len(self.down[i_level].attn) > 0:
                h = self.down[i_level].attn[i_block](h)
            hs.append(h)
        if i_level != self.num_resolutions - 1:
            hs.append(self.down[i_level].downsample(hs[-1]))
    # middle
    h = hs[-1]
    h = self.mid.block_1(h, temb)
    h = self.mid.attn_1(h)
    h = self.mid.block_2(h, temb)

    # end
    h = self.norm_out(h)
    h = swish(h)
    h = self.conv_out(h)
    return h

class Decoder:
  def __init__(self, config: EncoderDecoderConfig):
    self.config = config
    temb_ch = 0
    self.num_resolutions = len(config.ch_mult)

    in_ch_mult = (1,) + tuple(config.ch_mult)
    block_in = config.ch * config.ch_mult[self.num_resolutions-1]
    curr_res = config.resolution // 2 ** (self.num_resolutions - 1)
    print(f"Tokenizer : shape of latent is {config.z_channels, curr_res, curr_res}.")

    # z to block_in
    self.conv_in = nn.Conv2d(config.z_channels, block_in, 3, 1, 1)

    self.mid = SimpleNamespace()
    self.mid.block_1 = ResnetBlock(block_in, block_in, dropout=config.dropout, temb_channels=temb_ch)
    self.mid.attn_1 = ResAttnBlock(block_in)
    self.mid.block_2 = ResnetBlock(block_in, block_in, dropout=config.dropout, temb_channels=temb_ch)

    self.up = []
    for i_level in reversed(range(self.num_resolutions)):
      block = []
      attn = []
      block_out = config.ch * in_ch_mult[i_level]
      for _ in range(config.num_res_blocks+1):
        block.append(ResnetBlock(block_in, block_out, dropout=config.dropout, temb_channels=temb_ch))
        block_in = block_out
        if curr_res in config.attn_resolutions:
          attn.append(ResAttnBlock(block_in))
      up = SimpleNamespace()
      up.block = block
      up.attn = attn
      if i_level != 0:
        up.upsample = Upsample(block_in, with_conv=True)
        curr_res *= 2
      self.up.insert(0, up) # prepend to get the order right

    self.norm_out = nn.GroupNorm(num_groups=32, num_channels=block_in, eps=1e-6, affine=True)

    self.conv_out = nn.Conv2d(block_in, config.out_ch, 3, 1, 1)


  def __call__(self, z: Tensor) -> Tensor:
    temb = None
    h = self.conv_in(z)

    # middle
    h = self.mid.block_1(h, temb)
    h = self.mid.attn_1(h)
    h = self.mid.block_2(h, temb)

    # up
    for i_level in reversed(range(self.num_resolutions)):
      for i_block in range(self.config.num_res_blocks):
        h = self.up[i_level].block[i_block](h, temb)
        if len(self.up[i_level].attn) > 0:
          h = self.up[i_level].attn[i_block](h)
      if i_level != 0:
        h = self.up[i_level].upsample(h)

    h = self.norm_out(h)
    h = swish(h)
    h = self.conv_out(h)
    return h

# taken from "https://github.com/tinygrad/tinygrad/blob/master/examples/yolov8.py" (now 3 models use nearest upsampling)
def upsample(x: Tensor, scale_factor: float) -> Tensor:
  assert len(x.shape) > 2 and len(x.shape) <= 5
  (b, c), _lens = x.shape[:2], len(x.shape[2:])
  tmp = x.reshape([b, c, -1] + [1] * _lens) * Tensor.ones(*[1, 1, 1] + [scale_factor] * _lens)
  return tmp.reshape(list(x.shape) + [scale_factor] * _lens).permute([0, 1] + list(chain.from_iterable([[y+2, y+2+_lens] for y in range(_lens)]))).reshape([b, c] + [x * scale_factor for x in x.shape[2:]])

class Upsample:
  def __init__(self, in_channels:int, with_conv:bool) -> None:
    self.with_conv = with_conv
    if self.with_conv:
      self.conv = nn.Conv2d(in_channels, in_channels, 3, 1, 1)

  def __call__(self, x: Tensor) -> Tensor:
    if self.with_conv:
      return self.conv(upsample(x, scale_factor=2))

    return upsample(x, scale_factor=2)

class Downsample:
  def __init__(self, in_channels:int, with_conv:bool) -> None:
    self.with_conv = with_conv
    if self.with_conv:
      self.conv = nn.Conv2d(in_channels, in_channels, 3, 2, 0)

  def __call__(self, x: Tensor) -> Tensor:
    if self.with_conv:
      pad = (0, 1, 0, 1)
      return self.conv(x.pad2d(pad, 0))

    return x.avg_pool2d(kernel_size=(2, 2), stride=2)

class ResnetBlock:
  def __init__(self, in_channels:int, out_channels:Optional[int] = None, conv_shortcut:bool=False,
               dropout: float = 0., temb_channels: int = 512) -> None:
    self.in_channels = in_channels
    self.out_channels = in_channels if out_channels is None else out_channels
    self.use_conv_shortcut = conv_shortcut
    self.dropout = dropout

    self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
    self.norm1 = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
    if temb_channels > 0:
      self.temb_proj = nn.Linear(temb_channels, out_channels)
    self.norm2 = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
    self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
    if self.in_channels != self.out_channels:
      if self.use_conv_shortcut:
        self.conv_shortcut = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
      else:
        self.nin_shortcut = nn.Conv2d(in_channels, out_channels, 1, 1, 0)

  def __call__(self, x:Tensor, temb:Optional[Tensor]) -> Tensor:
    h = self.norm1(x)
    h = swish(h)
    h = self.conv1(h)

    if temb is not None:
      h = h + self.temb_proj(temb).reshape(h.shape)

    h = self.norm2(h)
    h = swish(h)
    h = h.dropout(self.dropout)
    h = self.conv2(h)
    if self.in_channels != self.out_channels:
      if self.use_conv_shortcut:
        x = self.conv_shortcut(x)
      else:
        x = self.nin_shortcut(x)
    return x + h

class ResAttnBlock:
  def __init__(self, embed_dim: int) -> None:
    self.norm = nn.GroupNorm(num_groups=32, num_channels=embed_dim, eps=1e-6, affine=True)

    self.query = (
      Tensor.scaled_uniform(embed_dim, embed_dim),
      Tensor.zeros(embed_dim),
    )
    self.key = (
      Tensor.scaled_uniform(embed_dim, embed_dim),
      Tensor.zeros(embed_dim),
    )
    self.value = (
      Tensor.scaled_uniform(embed_dim, embed_dim),
      Tensor.zeros(embed_dim),
    )

    self.out = (
      Tensor.scaled_uniform(embed_dim, embed_dim),
      Tensor.zeros(embed_dim),
    )

  def attn(self, x: Tensor) -> Tensor:
    query, key, value = [
      x.linear(*y)
      .transpose(1, 2)
      for y in [self.query, self.key, self.value]
    ]
    attention = Tensor.scaled_dot_product_attention(query, key, value).transpose(
      1, 2
    )
    return attention.linear(*self.out)

  def __call__(self, x: Tensor) -> Tensor:
    return x + self.attn(self.norm(x))

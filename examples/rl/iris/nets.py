from tinygrad import Tensor, nn
from typing import Optional
from itertools import chain

def swish(x: Tensor): return x.mul(x.sigmoid())

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

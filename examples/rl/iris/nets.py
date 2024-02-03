from tinygrad import Tensor, nn
from typing import Optional

def swish(x: Tensor): return x.mul(x.sigmoid())

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


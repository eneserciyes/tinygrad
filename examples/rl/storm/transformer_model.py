from typing import Callable, List
import tinygrad
from tinygrad import Tensor, nn

from attention_blocks import AttentionBlockKVCache, PositionalEncoding1D, AttentionBlock

class StochasticTransformer:
  def __init__(self, stoch_dim: int, action_dim:int, feat_dim: int, num_layers:int, num_heads:int, max_length:int, dropout:float):
    self.action_dim = action_dim

    self.stem1 = Tensor.scaled_uniform(stoch_dim+action_dim, feat_dim)
    self.stem2 = Tensor.scaled_uniform(feat_dim, feat_dim)
    self.position_encoding = PositionalEncoding1D(max_length, embed_dim=feat_dim)
    self.layer_stack = [AttentionBlock(feat_dim=feat_dim, hidden_dim=feat_dim*2, num_heads=num_heads, dropout=dropout) for _ in range(num_layers)]
    self.head = nn.Linear(feat_dim, stoch_dim)

  def __call__(self, samples: Tensor, action:Tensor, mask:Tensor):
    action = action.cast(tinygrad.dtypes.long).one_hot(self.action_dim).float()
    feats = samples.cat(action, dim=-1)
    feats = feats.linear(self.stem1).layernorm().relu().linear(self.stem2).layernorm()
    feats = self.position_encoding(feats).layernorm()

    for enc_layer in self.layer_stack:
      feats = enc_layer(feats, mask)

    return self.head(feats)

class StochasticTransformerKVCache:
  def __init__(self, stoch_dim: int, action_dim:int, feat_dim: int, num_layers: int, num_heads: int, max_length: int, dropout: float):
    self.action_dim = action_dim
    self.feat_dim = feat_dim

    self.stem: List[Callable] = [
      nn.Linear(stoch_dim+action_dim, feat_dim, bias=False),
      nn.LayerNorm(feat_dim),
      lambda x: x.relu(),
      nn.Linear(feat_dim, feat_dim, bias=False),
      nn.LayerNorm(feat_dim),
    ]
    self.position_encoding = PositionalEncoding1D(max_length, embed_dim=feat_dim)
    self.layer_stack = [
        AttentionBlockKVCache(feat_dim=feat_dim, hidden_dim=feat_dim*2, num_heads=num_heads, dropout=dropout) for _ in range(num_layers)
    ]
    self.layer_norm = nn.LayerNorm(feat_dim, eps=1e-6)
    self.kv_cache_list: List[Tensor] = []

  def __call__(self, samples:Tensor, action:Tensor, mask:Tensor):
    action = action.cast(tinygrad.dtypes.long).one_hot(self.action_dim).float()
    feats = samples.cat(action, dim=-1)
    feats = feats.sequential(self.stem)
    feats = self.position_encoding(feats).layernorm()

    for layer in self.layer_stack:
      feats = layer(feats, feats, feats, mask)

    return feats

  def reset_kv_cache_list(self, batch_size, dtype):
    self.kv_cache_list = []
    for _ in self.layer_stack:
      self.kv_cache_list.append(Tensor.zeros((batch_size, 0, self.feat_dim), dtype=dtype))

  def forward_with_kv_cache(self, samples, action):
    assert samples.shape[1] == 1
    mask = Tensor.ones((1, 1, self.kv_cache_list[0].shape[1]+1)) == 1

    action = action.cast(tinygrad.dtypes.long).one_hot(self.action_dim).float()
    feats = samples.cat(action, dim=-1)
    feats = feats.sequential(self.stem)
    feats = self.position_encoding.forward_with_position(feats, position=self.kv_cache_list[0].shape[1])
    feats = self.layer_norm(feats)

    for i, layer in enumerate(self.layer_stack):
      self.kv_cache_list[i] = self.kv_cache_list[i].cat(feats, dim=1)
      feats = layer(feats, self.kv_cache_list[i], self.kv_cache_list[i], mask)

    return feats


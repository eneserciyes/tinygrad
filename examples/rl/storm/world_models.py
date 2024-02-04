from tinygrad import Tensor, nn
import tinygrad

from transformer_model import StochasticTransformerKVCache
from distributions import Categorical, CategoricalKLDivLossWithFreeBits
from functions_losses import SymLogTwoHotLoss

class EncoderBN:
  def __init__(self, in_channels: int, stem_channels: int, final_feature_width: int) -> None:
    self.conv_in = nn.Conv2d(in_channels, stem_channels, 4,  2, 1, bias=False)
    self.bn_in = nn.BatchNorm2d(stem_channels)
    feature_width = 32
    channels = stem_channels

    self.convs = []
    self.bns = []
    while True:
      self.convs.append(nn.Conv2d(channels, channels*2, 4, 2, 1, bias=False))
      self.bns.append(nn.BatchNorm2d(stem_channels))
      channels *= 2
      feature_width //= 2
      if feature_width == final_feature_width:
        break

    self.last_channels = channels

  def __call__(self, x: Tensor) -> Tensor:
    B, L = x.shape[0], x.shape[1]
    x = x.reshape(B*L, *x.shape[2:])
    x = self.bn_in(self.conv_in(x)).relu()
    for (conv, bn) in zip(self.convs, self.bns):
      x = bn(conv(x)).relu()
    return x.reshape(B, L, *x.shape[1:])

class DecoderBN:
  def __init__(self, stoch_dim: int, last_channels: int, original_in_channels: int, stem_channels: int, final_feature_width: int) -> None:
    self.stem_in = Tensor.scaled_uniform(stoch_dim, last_channels*final_feature_width*final_feature_width)
    self.bn_in = nn.BatchNorm2d(last_channels)
    self.last_channels = last_channels
    self.final_feature_width = final_feature_width
    channels = last_channels
    feat_width = final_feature_width

    self.convs = []
    self.bns = []
    while True:
      if channels == stem_channels: break
      self.convs.append(nn.ConvTranspose2d(channels, channels//2, 4, 2, 1, bias=False))
      channels //= 2
      feat_width *= 2
      self.bns.append(nn.BatchNorm2d(channels))

    self.conv_out = nn.ConvTranspose2d(stem_channels, original_in_channels, 4, 2, 1)

  def __call__(self, sample: Tensor) -> Tensor:
    B, L = sample.shape[0], sample.shape[1]
    stem = sample.linear(self.stem_in).reshape(B*L, self.last_channels, self.final_feature_width, -1)
    obs_hat = self.bn_in(stem).relu()
    for (conv, bn) in zip(self.convs, self.bns):
      obs_hat = bn(conv(obs_hat)).relu()
    return self.conv_out(obs_hat)

class DistHead:
  def __init__(self, image_feat_dim: int, transformer_hidden_dim: int, stoch_dim: int) -> None:
    self.stoch_dim = stoch_dim
    self.post_head = nn.Linear(image_feat_dim, stoch_dim**2)
    self.prior_head = nn.Linear(transformer_hidden_dim, stoch_dim**2)

  def unimix(self, logits, mixing_ratio=0.01):
    probs = logits.softmax(-1)
    mixed_probs = mixing_ratio * Tensor.ones(*probs.shape) / self.stoch_dim + (1 - mixing_ratio) * probs
    return mixed_probs.log()

  def forward_post(self, x: Tensor) -> Tensor:
    logits = self.post_head(x)
    logits = logits.reshape(*logits.shape[:-1], self.stoch_dim, -1)
    return self.unimix(logits)

  def forward_prior(self, x:Tensor) -> Tensor:
    logits = self.prior_head(x)
    logits = logits.reshape(*logits.shape[:-1], self.stoch_dim, -1)
    return self.unimix(logits)

class RewardDecoder:
  def __init__(self, num_classes:int, transformer_hidden_dim:int) -> None:
    self.l1 = Tensor.scaled_uniform(transformer_hidden_dim, transformer_hidden_dim)
    self.l2 = Tensor.scaled_uniform(transformer_hidden_dim, transformer_hidden_dim)
    self.head = nn.Linear(transformer_hidden_dim, num_classes)

  def __call__(self,  feat:Tensor) -> Tensor:
    x = feat.linear(self.l1).layernorm().relu().linear(self.l2).layernorm().relu()
    return self.head(x)

class TerminationDecoder:
  def __init__(self, transformer_hidden_dim:int) -> None:
    self.l1 = Tensor.scaled_uniform(transformer_hidden_dim, transformer_hidden_dim)
    self.l2 = Tensor.scaled_uniform(transformer_hidden_dim, transformer_hidden_dim)
    self.head = nn.Linear(transformer_hidden_dim, 1)

  def __call__(self, feat:Tensor) -> Tensor:
    x = feat.linear(self.l1).layernorm().relu().linear(self.l2).layernorm().relu()
    return self.head(x).squeeze(-1)


class WorldModel:
  def __init__(self, in_channels: int, action_dim: int, transformer_max_length: int, transformer_hidden_dim: int,
               transformer_num_layers: int, transformer_num_heads: int) -> None:
    self.transformer_hidden_dim = transformer_hidden_dim
    self.final_feature_width = 4
    self.stoch_dim = 32
    self.stoch_flattened_dim = self.stoch_dim ** 2
    self.use_amp = False
    self.dtype = tinygrad.dtypes.bfloat16 if self.use_amp else tinygrad.dtypes.float32
    self.imagine_batch_size = -1
    self.imagine_batch_length = -1

    self.encoder = EncoderBN(in_channels, stem_channels=32, final_feature_width=self.final_feature_width)
    self.storm_transformer = StochasticTransformerKVCache(
        stoch_dim=self.stoch_flattened_dim, action_dim=action_dim, feat_dim=transformer_hidden_dim,
        num_layers=transformer_num_layers, num_heads=transformer_num_heads, max_length=transformer_max_length, dropout=0.1)
    self.dist_head = DistHead(image_feat_dim=self.encoder.last_channels*(self.final_feature_width**2),
                              transformer_hidden_dim=transformer_hidden_dim, stoch_dim=self.stoch_dim)
    self.image_decoder = DecoderBN(stoch_dim=self.stoch_flattened_dim, last_channels=self.encoder.last_channels,
                                   original_in_channels=in_channels, stem_channels=32, final_feature_width=self.final_feature_width)
    self.reward_decoder = RewardDecoder(num_classes=255, transformer_hidden_dim=transformer_hidden_dim)
    self.termination_decoder = TerminationDecoder(transformer_hidden_dim=transformer_hidden_dim)

    self.symlog_twohot_loss_func = SymLogTwoHotLoss(num_classes=255, lower_bound=-20, upper_bound=20)
    self.categorical_kl_div_loss = CategoricalKLDivLossWithFreeBits(free_bits=1)

  def encode_obs(self, obs: Tensor) -> Tensor:
    embedding = self.encoder(obs)
    post_logits = self.dist_head.forward_post(embedding)
    sample = self.straight_through_gradient(post_logits)
    return sample.flatten(2)

  def straight_through_gradient(self, logits: Tensor) -> Tensor:
    dist = Categorical(logits=logits)
    return dist.onehot_sample() + dist.probs - dist.probs.detach()


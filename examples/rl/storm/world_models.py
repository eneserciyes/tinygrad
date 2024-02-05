from tinygrad import Tensor, nn
import tinygrad

from transformer_model import StochasticTransformerKVCache
from distributions import Categorical, CategoricalKLDivLossWithFreeBits
from functions_losses import SymLogTwoHotLoss
import agents

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

    # TODO: add optimizer here

  def encode_obs(self, obs: Tensor) -> Tensor:
    embedding = self.encoder(obs)
    post_logits = self.dist_head.forward_post(embedding)
    sample = self.straight_through_gradient(post_logits)
    return sample.flatten(2)

  def straight_through_gradient(self, logits: Tensor) -> Tensor:
    dist = Categorical(logits=logits)
    return dist.onehot_sample() + dist.probs - dist.probs.detach()

  def calc_last_dist_feat(self, latent: Tensor, action: Tensor):
    temporal_mask = (1 - Tensor.triu(Tensor.ones(1, latent.shape[1], latent.shape[1]), k=1)) == 1
    dist_feat = self.storm_transformer(latent, action, temporal_mask)
    last_dist_feat = dist_feat[:, -1:]
    prior_logits = self.dist_head.forward_prior(last_dist_feat)
    prior_sample = self.straight_through_gradient(prior_logits)
    prior_flattened_sample = prior_sample.flatten(2)
    return prior_flattened_sample, last_dist_feat

  def predict_next(self, last_flattened_sample, action, log_video=True):
    dist_feat = self.storm_transformer.forward_with_kv_cache(last_flattened_sample, action)
    prior_logits = self.dist_head.forward_prior(dist_feat)
    # decoding
    prior_sample = self.straight_through_gradient(prior_logits)
    prior_flattened_sample = prior_sample.flatten(2)
    if log_video:
        obs_hat = self.image_decoder(prior_flattened_sample)
    else:
        obs_hat = None
    reward_hat = self.reward_decoder(dist_feat)
    reward_hat = self.symlog_twohot_loss_func.decode(reward_hat)
    termination_hat = self.termination_decoder(dist_feat)
    termination_hat = termination_hat > 0
    return obs_hat, reward_hat, termination_hat, prior_flattened_sample, dist_feat

  def init_imagine_buffer(self, imagine_batch_size, imagine_batch_length, dtype):
    print(f"init imagine_buffer: {imagine_batch_size}x{imagine_batch_length}@{dtype}")
    self.imagine_batch_size = imagine_batch_size
    self.imagine_batch_length = imagine_batch_length
    latent_size = (imagine_batch_size, imagine_batch_length+1, self.stoch_flattened_dim)
    hidden_size = (imagine_batch_size, imagine_batch_length+1, self.transformer_hidden_dim)
    scalar_size = (imagine_batch_size, imagine_batch_length)
    self.latent_buffer = Tensor.zeros(*latent_size, dtype=dtype)
    self.hidden_buffer = Tensor.zeros(*hidden_size, dtype=dtype)
    self.action_buffer = Tensor.zeros(*scalar_size, dtype=dtype)
    self.reward_hat_buffer = Tensor.zeros(*scalar_size, dtype=dtype)
    self.termination_hat_buffer = Tensor.zeros(*scalar_size, dtype=dtype)

  def imagine_data(self, agent: agents.ActorCriticAgent, sample_obs, sample_action,
                   imagine_batch_size, imagine_batch_length, log_video, logger):
    self.init_imagine_buffer(imagine_batch_size, imagine_batch_length, dtype=self.dtype)
    obs_hat_list = []
    self.storm_transformer.reset_kv_cache_list(imagine_batch_size, dtype=self.dtype)
    context_latent = self.encode_obs(sample_obs)
    for i in range(sample_obs.shape[1]):
      last_obs_hat, last_reward_hat, last_termination_hat, last_latent, last_dist_feat = self.predict_next(
        context_latent[:, i:i+1], sample_action[:, i:i+1], log_video=log_video)
    self.latent_buffer[:, 0:1] = last_latent
    self.hidden_buffer[:, 0:1] = last_dist_feat

    # imagine
    for i in range(imagine_batch_length):
      action = agent.sample(Tensor.cat(self.latent_buffer[:, i:i+1], self.hidden_buffer[:, i:i+1], dim=-1))
      self.action_buffer[:, i:i+1] = action
      last_obs_hat, last_reward_hat, last_termination_hat, last_latent, last_dist_feat = self.predict_next(
        self.latent_buffer[:, i:i+1], self.action_buffer[:, i:i+1], log_video=log_video)

      self.latent_buffer[:, i+1:i+2] = last_latent
      self.hidden_buffer[:, i+1:i+2] = last_dist_feat
      self.reward_hat_buffer[:, i:i+1] = last_reward_hat
      self.termination_hat_buffer[:, i:i+1] = last_termination_hat
      if log_video:
        obs_hat_list.append(last_obs_hat[::imagine_batch_size//16])

    if log_video:
      logger.log("Imagine/predict_video", obs_hat_list[0].cat(*obs_hat_list[1:], dim=1).clip(0, 1).float().detach().numpy())

    return self.latent_buffer.cat(self.hidden_buffer, dim=-1), self.action_buffer, self.reward_hat_buffer, self.termination_hat_buffer

  def update(self, obs: Tensor, action: Tensor, reward: Tensor, termination: Tensor, logger=None):
    _, batch_length = obs.shape[:2]
    embedding = self.encoder(obs)
    post_logits = self.dist_head.forward_post(embedding)
    sample = self.straight_through_gradient(post_logits)
    flattened_sample = sample.flatten(2)

    # decoding image
    obs_hat = self.image_decoder(flattened_sample)

    # transformer
    temporal_mask = (1 - Tensor.triu(Tensor.ones(1, batch_length, batch_length), k=1)) == 1
    dist_feat = self.storm_transformer(flattened_sample, action, temporal_mask)
    prior_logits = self.dist_head.forward_prior(dist_feat)
    reward_hat = self.reward_decoder(dist_feat)
    termination_hat = self.termination_decoder(dist_feat)

    # env loss
    reconstruction_loss = (obs_hat - obs).sum(axis=(2,3,4)).pow(2).mean()
    reward_loss = self.symlog_twohot_loss_func(reward_hat, reward)
    # BCE with logits loss
    termination_loss = (termination.float() * termination_hat.sigmoid().log() + (1 - termination).float() * (1 - termination_hat.sigmoid()).log()).mean()

    # dyn-rep loss
    dynamics_loss, dynamics_real_kl_div = self.categorical_kl_div_loss(post_logits[:, 1:].detach(), prior_logits[:, :-1])
    representation_loss, representation_real_kl_div = self.categorical_kl_div_loss(post_logits[:, 1:], prior_logits[:, :-1].detach())
    total_loss = reconstruction_loss + reward_loss + termination_loss + 0.5 * dynamics_loss + 0.1 * representation_loss

    total_loss.backward()

    # TODO: check if grad scaler exists in Tinygrad.
    # TODO: check how to optimize
    if logger is not None:
      logger.log("WorldModel/reconstruction_loss", reconstruction_loss.item())
      logger.log("WorldModel/reward_loss", reward_loss.item())
      logger.log("WorldModel/termination_loss", termination_loss.item())
      logger.log("WorldModel/dynamics_loss", dynamics_loss.item())
      logger.log("WorldModel/dynamics_real_kl_div", dynamics_real_kl_div.item())
      logger.log("WorldModel/representation_loss", representation_loss.item())
      logger.log("WorldModel/representation_real_kl_div", representation_real_kl_div.item())
      logger.log("WorldModel/total_loss", total_loss.item())






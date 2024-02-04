from dataclasses import dataclass
from typing import Optional

from tinygrad import Tensor, nn
from extra.models.rnnt import LSTMCell

from tokenizer import Batch, Tokenizer, Loss

def compute_lambda_returns(rewards, values, ends, gamma, lambda_):
  assert rewards.ndim == 2 or (rewards.ndim == 3 and rewards.size(2) == 1)
  assert rewards.shape == ends.shape == values.shape, f"{rewards.shape}, {values.shape}, {ends.shape}"  # (B, T, 1)
  t = rewards.size(1)
  lambda_returns = Tensor.empty(*values.shape)
  lambda_returns[:, -1] = values[:, -1]
  lambda_returns[:, :-1] = rewards[:, :-1] + ends[:, :-1].logical_not() * gamma * (1 - lambda_) * values[:, 1:]

  last = values[:, -1]
  for i in list(range(t - 1))[::-1]:
      lambda_returns[:, i] += ends[:, i].logical_not() * gamma * lambda_ * last
      last = lambda_returns[:, i]

  return lambda_returns

@dataclass
class ActorCriticOutput:
  logits_actions: Tensor
  means_values: Tensor

@dataclass
class ImagineOutput:
  observations: Tensor
  actions: Tensor
  logits_actions: Tensor
  values: Tensor
  rewards: Tensor
  ends: Tensor

class ActorCritic:
  def __init__(self, act_vocab_size, use_original_obs: bool = False) -> None:
    self.use_original_obs = use_original_obs
    self.convs = [
        nn.Conv2d(3, 32, 3, 1, 1),
        nn.Conv2d(32, 32, 3, 1, 1),
        nn.Conv2d(32, 64, 3, 1, 1),
        nn.Conv2d(64, 64, 3, 1, 1),
    ]

    self.lstm_dim = 512
    self.lstm = LSTMCell(1024, self.lstm_dim, dropout=0.0)
    self.hx, self.cx = None, None

    self.critic_linear = nn.Linear(self.lstm_dim, 1)
    self.actor_linear = nn.Linear(self.lstm_dim, act_vocab_size)

  def clear(self) -> None:
    self.hx = None
    self.cx = None

  def reset(self, n:int, burnin_observations: Optional[Tensor] = None, mask_padding: Optional[Tensor] = None) -> None:
    self.hx = Tensor.zeros(n, self.lstm_dim)
    self.cx = Tensor.zeros(n, self.lstm_dim)
    if burnin_observations is not None:
      assert burnin_observations.ndim == 5 and burnin_observations.shape[0] == n and mask_padding is not None and burnin_observations.shape[:2] == mask_padding.shape
      for i in range(burnin_observations.shape[1]):
        if mask_padding[:, i].any():
          burnin_observations.requires_grad = False
          mask_padding.requires_grad = False
          self(burnin_observations[:, i], mask_padding[:, i])

  def prune(self, mask: Tensor) -> None:
    assert self.hx is not None and self.cx is not None
    self.hx = self.hx[mask]
    self.cx = self.cx[mask]

  def __call__(self, inputs: Tensor, mask_padding: Optional[Tensor] = None) -> ActorCriticOutput:
    assert inputs.ndim == 4 and inputs.shape[1:] == (3, 64, 64)
    assert 0 <= inputs.min() <= 1 and 0 <= inputs.max() <= 1
    assert mask_padding is None or (mask_padding.ndim == 1 and mask_padding.shape[0] == inputs.shape[0] and mask_padding.any())
    x = inputs[mask_padding] if mask_padding is not None else inputs
    x = x.mul(2).sub(1)
    for conv in self.convs:
      x = conv(x).max_pool2d(kernel_size=(2,2)).relu()
    x = x.reshape(x.shape[0], -1)

    if mask_padding is None:
      if self.hx is None or self.cx is None:
        self.hx = Tensor.zeros(x.shape[0], self.lstm_dim)
        self.cx = Tensor.zeros(x.shape[0], self.lstm_dim)
      self.hx, self.cx = self.lstm(x, Tensor.cat(self.hx, self.cx))
    else:
      assert self.hx is not None and self.cx is not None
      self.hx[mask_padding], self.cx[mask_padding] = self.lstm(x, Tensor.cat(self.hx[mask_padding], self.cx[mask_padding]))

    logits_actions = self.actor_linear(self.hx).unsqueeze(1)
    means_values = self.critic_linear(self.hx).unsqueeze(-1)
    return ActorCriticOutput(logits_actions, means_values)

  def compute_loss(self, batch: Batch, tokenizer: Tokenizer, world_model: WorldModel, imagine_horizon: int, gamma: float, lambda_: float, entropy_weight: float) -> Loss:
    assert not self.use_original_obs
    outputs = self.imagine(batch, tokenizer, world_model, horizon=imagine_horizon)

    lambda_returns = compute_lambda_returns(outputs.rewards, outputs.values, outputs.ends, gamma, lambda_)[:, :-1]
    lambda_returns.requires_grad = False
    values = outputs.values[:, :-1]
    # TODO: implement Categorical log_prob in TinyGrad
    return Loss()

  def imagine(self, batch: Batch, tokenizer: Tokenizer, world_model: WorldModel, horizon: int, show_pbar: bool = False) -> ImagineOutput:
    assert not self.use_original_obs
    initial_observations = batch['observations']
    mask_padding = batch['mask_padding']
    assert initial_observations.ndim == 5 and initial_observations.shape[2:] == (3, 64, 64)
    assert mask_padding[:, -1].all()

import copy
from examples.rl.storm.distributions import Categorical
from functions_losses import SymLogTwoHotLoss
from tinygrad import Tensor, nn
from tinygrad.nn.optim import Adam
from tinygrad.nn.state import get_parameters
import tinygrad


class EMAScalar:
    def __init__(self, decay) -> None:
        self.scalar = 0.0
        self.decay = decay

    def __call__(self, value):
        self.update(value)
        return self.get()

    def update(self, value):
        self.scalar = self.scalar * self.decay + value * (1 - self.decay)

    def get(self):
        return self.scalar

def percentile(x:Tensor, percentage):
  flat_x = x.flatten()
  kth = int(flat_x.shape[0] * percentage)
  # TODO: implement kth value
  # TODO: implement quickselect for Tensor
  return flat_x[kth]

def calc_lambda_return(rewards, values, termination, gamma, lam, dtype=tinygrad.dtypes.float32):
  inv_termination = 1 - termination
  batch_size, batch_length = rewards.shape[:2]
  gamma_return = Tensor.zeros(batch_size, batch_length+1, dtype=dtype)
  gamma_return[:, -1] = values[:, -1] # TODO: check if this is correct
  for t in reversed(range(batch_length)):
    gamma_return[:, t] = rewards[:, t] + \
        gamma * inv_termination[:, t] * (1-lam) * values[:, t] + \
        gamma * inv_termination[:, t] * lam * gamma_return[:, t+1]

  return gamma_return[:, :-1]

class ActorCriticAgent:
  def __init__(self, feat_dim, num_layers, hidden_dim, action_dim, gamma, lambd, entropy_coef) -> None:
    self.gamma = gamma
    self.lambd = lambd
    self.entropy_coef = entropy_coef
    self.use_amp = False
    self.tensor_dtype = tinygrad.dtypes.bfloat16 if self.use_amp else tinygrad.dtypes.float32

    self.symlog_twohot_loss = SymLogTwoHotLoss(255, -20, 20)

    self.actor = [
      nn.Linear(feat_dim, hidden_dim, bias=False)
    ]
    for _ in range(num_layers-1):
      self.actor.append(nn.Linear(hidden_dim, hidden_dim, bias=False))
    self.action_out = nn.Linear(hidden_dim, action_dim)
    self.critic = [
      nn.Linear(feat_dim, hidden_dim, bias=False)
    ]
    for _ in range(num_layers-1):
      self.critic.append(nn.Linear(hidden_dim, hidden_dim, bias=False))
    self.value_out = nn.Linear(hidden_dim, 255)

    self.slow_critic = copy.deepcopy(self.critic)
    self.slow_value_out = copy.deepcopy(self.value_out)

    self.lowerbound_ema = EMAScalar(decay=0.99)
    self.upperbound_ema = EMAScalar(decay=0.99)

    self.optimizer = Adam(get_parameters(self), lr=3e-5, eps=1e-5)

  def update_slow_critic(self, decay=0.98):
    for slow_param, param in zip(get_parameters(self.slow_critic), self.critic):
      # TODO: find how to copy the slow_param to param
      pass
    # TODO: copy slow_value_out to value_out

  def policy(self, x:Tensor):
    for layer in self.actor:
      x = layer(x).layernorm().relu()
    logits = self.action_out(x)
    return logits

  def critic_fn(self, x:Tensor):
    for layer in self.critic:
      x = layer(x).layernorm().relu()
    value = self.value_out(x)
    return value

  def value(self, x):
    value = self.critic_fn(x)
    return self.symlog_twohot_loss.decode(value)

  def slow_value(self, x:Tensor):
    x.requires_grad = False
    for layer in self.slow_critic:
      x = layer(x).layernorm().relu()
    value = self.slow_value_out(x)
    return self.symlog_twohot_loss.decode(value)

  def get_logits_raw_value(self, x):
    return self.policy(x), self.critic_fn(x)

  def sample(self, latent: Tensor, greedy=False):
    latent.requires_grad = False # TODO: check if this is enough
    # TODO: autocast
    logits = self.policy(latent)
    dist = Categorical(logits)
    action = dist.probs.argmax(-1) if greedy else dist.sample()
    return action

  def sample_as_env_action(self, latent: Tensor, greedy=False):
    action = self.sample(latent, greedy)
    return action.detach().squeeze(-1).numpy()

  def update(self, latent: Tensor, action: Tensor, reward: Tensor, termination: Tensor, logger=None):
    with Tensor.train():
      # TODO: autocast
      logits, raw_value = self.get_logits_raw_value(latent)
      dist = Categorical(logits[:, :-1])
      log_prob = dist.log_prob(action) # TODO: implement log_probs for dist
      entropy = dist.entropy() # TODO: implement entropy for dist

      slow_value = self.slow_value(latent)
      slow_lambda_return = calc_lambda_return(reward, slow_value, termination, self.gamma, self.lambd)
      value = self.symlog_twohot_loss.decode(raw_value)
      lambda_return = calc_lambda_return(reward, value, termination, self.gamma, self.lambd)

      value_loss = self.symlog_twohot_loss(raw_value[:, :-1], lambda_return.detach())
      slow_value_regularization_loss = self.symlog_twohot_loss(raw_value[:, :-1], slow_lambda_return.detach())

      lower_bound = self.lowerbound_ema(percentile(lambda_return, 0.05))
      upper_bound = self.upperbound_ema(percentile(lambda_return, 0.95))
      S = upper_bound - lower_bound
      norm_ratio = S.max(Tensor.ones(1))
      norm_advantage = (lambda_return - value[:, :-1]) / norm_ratio
      policy_loss = -(log_prob * norm_advantage.detach()).mean()

      entropy_loss = entropy.mean()

      loss = policy_loss + value_loss + slow_value_regularization_loss - self.entropy_coef * entropy_loss

    loss.backward() # TODO: grad scaler if autocast is used
    # TODO: clip gradients
    self.optimizer.step()
    self.optimizer.zero_grad()

    self.update_slow_critic()

    if logger is not None:
      logger.log('ActorCritic/policy_loss', policy_loss.item())
      logger.log('ActorCritic/value_loss', value_loss.item())
      logger.log('ActorCritic/entropy_loss', entropy_loss.item())
      logger.log('ActorCritic/S', S.item())
      logger.log('ActorCritic/norm_ratio', norm_ratio.item())
      logger.log('ActorCritic/total_loss', loss.item())


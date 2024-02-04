import pickle

from tinygrad import Tensor
import tinygrad

class ReplayBuffer:
  def __init__(self, obs_shape, num_envs, max_length=int(1e6), warmup_length=50000) -> None:
    self.obs_buffer = Tensor.empty(max_length//num_envs, num_envs, *obs_shape, dtype=tinygrad.dtypes.uint8)
    self.action_buffer = Tensor.empty(max_length//num_envs, num_envs, dtype=tinygrad.dtypes.float32)
    self.reward_buffer = Tensor.empty(max_length//num_envs, num_envs, dtype=tinygrad.dtypes.float32)
    self.termination_buffer = Tensor.empty(max_length//num_envs, num_envs, dtype=tinygrad.dtypes.float32)
    self.obs_buffer.requires_grad = False
    self.action_buffer.requires_grad = False
    self.reward_buffer.requires_grad = False
    self.termination_buffer.requires_grad = False

    self.length = 0
    self.num_envs = num_envs
    self.last_pointer = -1
    self.max_length = max_length
    self.warmup_length = warmup_length
    self.external_buffer_length = None

  def load_trajectory(self, path):
    buffer = pickle.load(open(path, "rb"))
    self.external_buffer = {name: Tensor(buffer[name]) for name in buffer}
    self.external_buffer_length = self.external_buffer["obs"].shape[0]

  def sample_external(self, batch_size, batch_length):
    assert self.external_buffer_length is not None
    indices = Tensor.randint(batch_size, low=0, high=self.external_buffer_length+1-batch_length)
    obs = Tensor.stack([self.external_buffer["obs"][idx:idx+batch_length] for idx in indices])
    action = Tensor.stack([self.external_buffer["action"][idx:idx+batch_length] for idx in indices])
    reward = Tensor.stack([self.external_buffer["reward"][idx:idx+batch_length] for idx in indices])
    termination = Tensor.stack([self.external_buffer["termination"][idx:idx+batch_length] for idx in indices])
    return obs, action, reward, termination

  def ready(self):
    return self.length * self.num_envs > self.warmup_length

  def sample(self, batch_size, external_batch_size, batch_length):
    obs, action, reward, termination = [], [], [], []
    if batch_size > 0:
      for i in range(self.num_envs):
        indices = Tensor.randint(batch_size // self.num_envs, low=0, high=self.length+1-batch_length)
        obs.append(Tensor.stack([self.obs_buffer[idx:idx+batch_length, i] for idx in indices]))
        action.append(Tensor.stack([self.action_buffer[idx:idx+batch_length, i] for idx in indices]))
        reward.append(Tensor.stack([self.reward_buffer[idx:idx+batch_length, i] for idx in indices]))
        termination.append(Tensor.stack([self.termination_buffer[idx:idx+batch_length, i] for idx in indices]))
    if self.external_buffer_length is not None and external_batch_size > 0:
      external_obs, external_action, external_reward, external_termination = self.sample_external(external_batch_size, batch_length)
      obs.append(external_obs)
      action.append(external_action)
      reward.append(external_reward)
      termination.append(external_termination)

    obs = obs[0].cat(*obs[1:], dim=0).float() / 255
    obs = obs.permute((0, 1, 4, 2, 3))
    action = action[0].cat(*action[1:], dim=0)
    reward = reward[0].cat(*reward[1:], dim=0)
    termination = termination[0].cat(*termination[1:], dim=0)

    return obs, action, reward, termination

  def append(self, obs, action, reward, termination):
    self.last_pointer = (self.last_pointer + 1) % (self.max_length // self.num_envs)
    self.obs_buffer[self.last_pointer] = Tensor(obs)
    self.action_buffer[self.last_pointer] = Tensor(action)
    self.reward_buffer[self.last_pointer] = Tensor(reward)
    self.termination_buffer[self.last_pointer] = Tensor(termination)

    if len(self) < self.max_length:
      self.length += 1

  def __len__(self):
    return self.length * self.num_envs

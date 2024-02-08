import pickle

from tinygrad import Tensor
import numpy as np

class ReplayBuffer:
  def __init__(self, obs_shape, num_envs, max_length=int(1e6), warmup_length=50000) -> None:
    self.obs_buffer = np.empty((max_length//num_envs, num_envs, *obs_shape), dtype=np.uint8)
    self.action_buffer = np.empty((max_length//num_envs, num_envs), dtype=np.float32)
    self.reward_buffer = np.empty((max_length//num_envs, num_envs), dtype=np.float32)
    self.termination_buffer = np.empty((max_length//num_envs, num_envs), dtype=np.float32)

    self.length = 0
    self.num_envs = num_envs
    self.last_pointer = -1
    self.max_length = max_length
    self.warmup_length = warmup_length
    self.external_buffer_length = None

  def load_trajectory(self, path):
    buffer = pickle.load(open(path, "rb"))
    self.external_buffer = buffer
    self.external_buffer_length = self.external_buffer["obs"].shape[0]

  def sample_external(self, batch_size, batch_length):
    assert self.external_buffer_length is not None
    indices = np.random.randint(0, self.external_buffer_length+1-batch_length, size=batch_size)
    obs = np.stack([self.external_buffer["obs"][idx:idx+batch_length] for idx in indices])
    action = np.stack([self.external_buffer["action"][idx:idx+batch_length] for idx in indices])
    reward = np.stack([self.external_buffer["reward"][idx:idx+batch_length] for idx in indices])
    termination = np.stack([self.external_buffer["done"][idx:idx+batch_length] for idx in indices])

    return obs, action, reward, termination

  def ready(self):
    return self.length * self.num_envs > self.warmup_length

  def sample(self, batch_size, external_batch_size, batch_length):
    obs, action, reward, termination = [], [], [], []
    if batch_size > 0:
      for i in range(self.num_envs):
        indices = np.random.randint(0, self.length+1-batch_length, size=batch_size//self.num_envs)
        obs.append(np.stack([self.obs_buffer[idx:idx+batch_length, i] for idx in indices]))
        action.append(np.stack([self.action_buffer[idx:idx+batch_length, i] for idx in indices]))
        reward.append(np.stack([self.reward_buffer[idx:idx+batch_length, i] for idx in indices]))
        termination.append(np.stack([self.termination_buffer[idx:idx+batch_length, i] for idx in indices]))

    # TODO: make sure this doesn't get used and delete it
    if self.external_buffer_length is not None and external_batch_size > 0:
      external_obs, external_action, external_reward, external_termination = self.sample_external(external_batch_size, batch_length)
      obs.append(external_obs)
      action.append(external_action)
      reward.append(external_reward)
      termination.append(external_termination)

    obs = Tensor(np.concatenate(obs, axis=0)).float() / 255
    obs = obs.permute((0, 1, 4, 2, 3))
    action = Tensor(np.concatenate(action, axis=0))
    reward = Tensor(np.concatenate(reward, axis=0))
    termination = Tensor(np.concatenate(termination, axis=0))

    return obs, action, reward, termination

  def append(self, obs, action, reward, termination):
    self.last_pointer = (self.last_pointer + 1) % (self.max_length // self.num_envs)
    self.obs_buffer[self.last_pointer] = obs
    self.action_buffer[self.last_pointer] = action
    self.reward_buffer[self.last_pointer] = reward
    self.termination_buffer[self.last_pointer] = termination

    if len(self) < self.max_length:
      self.length += 1

  def __len__(self):
    return self.length * self.num_envs

import os
import colorama
from collections import deque
from tqdm import tqdm
import warnings
import argparse
import shutil

import gymnasium
import numpy as np
from omegaconf import OmegaConf

import env_wrapper
from replay_buffer import ReplayBuffer
from world_models import WorldModel
import agents

from tinygrad import Tensor
from tinygrad.nn.state import safe_save, get_state_dict

def build_single_env(env_name, image_size, seed):
  env = gymnasium.make(env_name, full_action_space=False, render_mode="rgb_array", frameskip=1)
  env = env_wrapper.SeedEnvWrapper(env, seed=seed)
  env = env_wrapper.MaxLast2FrameSkipWrapper(env, skip=4)
  env = gymnasium.wrappers.ResizeObservation(env, shape=image_size)
  env = env_wrapper.LifeLossInfo(env)
  return env

def build_vec_env(env_name, image_size, num_envs, seed):
  # lambda pitfall refs to: https://python.plainenglish.io/python-pitfalls-with-variable-capture-dcfc113f39b7
  def lambda_generator(env_name, image_size):
      return lambda: build_single_env(env_name, image_size, seed)
  env_fns = []
  env_fns = [lambda_generator(env_name, image_size) for _ in range(num_envs)]
  vec_env = gymnasium.vector.AsyncVectorEnv(env_fns=env_fns)
  return vec_env

def train_world_model_step(replay_buffer: ReplayBuffer, world_model: WorldModel, batch_size, demonstration_batch_size, batch_length, logger):
  obs, action, reward, termination = replay_buffer.sample(batch_size, demonstration_batch_size, batch_length)
  with Tensor.train():
    world_model.update(obs, action, reward, termination, logger=logger)

def world_model_imagine_data(replay_buffer: ReplayBuffer, world_model: WorldModel, agent: agents.ActorCriticAgent,
                             imagine_batch_size, imagine_demonstration_batch_size, imagine_context_length,
                             imagine_batch_length, log_video, logger):
  sample_obs, sample_action, _, _ = replay_buffer.sample(
      imagine_batch_size, imagine_demonstration_batch_size, imagine_context_length)
  latent, action, reward_hat, termination_hat = world_model.imagine_data(
      agent, sample_obs, sample_action,
      imagine_batch_size=imagine_batch_size+imagine_demonstration_batch_size,
      imagine_batch_length=imagine_batch_length,
      log_video=log_video, logger=logger)
  return latent, action, None, None, reward_hat, termination_hat

def joint_train_world_model_agent(env_name, max_steps, num_envs, image_size,
                                  replay_buffer: ReplayBuffer,
                                  world_model: WorldModel, agent: agents.ActorCriticAgent,
                                  train_dynamics_every_steps, train_agent_every_steps,
                                  batch_size, demonstration_batch_size, batch_length,
                                  imagine_batch_size, imagine_demonstration_batch_size, imagine_context_length,
                                  imagine_batch_length, save_every_steps, seed, logger):
  # create ckpt dir
  os.makedirs(f"ckpt/{args.n}", exist_ok=True)
  # create env
  vec_env = build_vec_env(env_name, image_size, num_envs=num_envs, seed=seed)
  print("Current env: ", colorama.Fore.YELLOW + env_name + colorama.Style.RESET_ALL)

  # reset envs and variables
  sum_reward = np.zeros(num_envs)
  current_obs, current_info = vec_env.reset()
  context_obs = deque(maxlen=16)
  context_action = deque(maxlen=16)

  for total_steps in tqdm(range(max_steps//num_envs)):
    if replay_buffer.ready():
      # no grad here
      if len(context_action) == 0:
        action = vec_env.action_space.sample()
      else:
        context_latent = world_model.encode_obs(context_obs[0].cat(list(context_obs)[1:], dim=1))
        model_context_action = Tensor(np.stack(list(context_action), axis=1))
        prior_flattened_sample, last_dist_feat = world_model.calc_last_dist_feat(context_latent, model_context_action)
        action = agent.sample_as_env_action(
            prior_flattened_sample.cat(last_dist_feat, dim=1), greedy=False)

      context_obs.append(Tensor(current_obs).permute(0, 3, 1, 2).unsqueeze(1) / 255)
      context_action.append(action)
    else:
      action = vec_env.action_space.sample()

    obs, reward, done, truncated, info = vec_env.step(action)
    replay_buffer.append(current_obs, action, reward, np.logical_or(done, info["life_loss"]))

    done_flag = np.logical_or(done, truncated)
    if done_flag.any():
      for i in range(num_envs):
        if done_flag[i]:
          if logger is not None:
            logger.log(f"sample/{env_name}_reward", sum_reward[i])
            logger.log(f"sample/{env_name}_episode_steps", current_info["episode_frame_number"][i]//4)  # framskip=4
            logger.log("replay_buffer/length", len(replay_buffer))
          sum_reward[i] = 0
    # update current_obs, current_info and sum_reward
    sum_reward += reward
    current_obs = obs
    current_info = info

    # train world model part >>>
    if replay_buffer.ready() and total_steps % (train_dynamics_every_steps//num_envs) == 0:
      train_world_model_step(
        replay_buffer=replay_buffer,
        world_model=world_model,
        batch_size=batch_size,
        demonstration_batch_size=demonstration_batch_size,
        batch_length=batch_length,
        logger=logger
      )

    # train agent part >>>
    if replay_buffer.ready() and total_steps % (train_agent_every_steps//num_envs) == 0 and total_steps*num_envs >= 0:
      if total_steps % (save_every_steps//num_envs) == 0:
        log_video = True
      else:
        log_video = False

      imagine_latent, agent_action, agent_logprob, agent_value, imagine_reward, imagine_termination = world_model_imagine_data(
        replay_buffer=replay_buffer, world_model=world_model, agent=agent, imagine_batch_size=imagine_batch_size,
        imagine_demonstration_batch_size=imagine_demonstration_batch_size,
        imagine_context_length=imagine_context_length, imagine_batch_length=imagine_batch_length,
        log_video=log_video, logger=logger
      )

      agent.update(latent=imagine_latent, action=agent_action, reward=imagine_reward, termination=imagine_termination, logger=logger)

    if total_steps % (save_every_steps//num_envs) == 0:
      print(colorama.Fore.GREEN + f"Saving model at total steps {total_steps}" + colorama.Style.RESET_ALL)
      safe_save(get_state_dict(world_model), f"ckpt/{args.n}/world_model_{total_steps}.safetensors")
      safe_save(get_state_dict(agent), f"ckpt/{args.n}/agent_{total_steps}.pth")


def build_world_model(conf, action_dim):
    return WorldModel(
        in_channels=conf.models.world_model.in_channels,
        action_dim=action_dim,
        transformer_max_length=conf.models.world_model.transformer_max_length,
        transformer_hidden_dim=conf.models.world_model.transformer_hidden_dim,
        transformer_num_layers=conf.models.world_model.transformer_num_layers,
        transformer_num_heads=conf.models.world_model.transformer_num_heads
    )

def build_agent(conf, action_dim):
    return agents.ActorCriticAgent(
        feat_dim=32*32+conf.models.world_model.transformer_hidden_dim,
        num_layers=conf.models.agent.num_layers,
        hidden_dim=conf.models.agent.hidden_dim,
        action_dim=action_dim,
        gamma=conf.models.agent.gamma,
        lambd=conf.models.agent.lambd,
        entropy_coef=conf.models.agent.entropy_coef,
    )

if __name__ == "__main__":
  warnings.filterwarnings('ignore')

  parser = argparse.ArgumentParser()
  parser.add_argument("-n", type=str, required=True)
  parser.add_argument("-seed", type=int, required=True)
  parser.add_argument("-config_path", type=str, required=True)
  parser.add_argument("-env_name", type=str, required=True)
  parser.add_argument("-trajectory_path", type=str, required=True)
  args = parser.parse_args()
  conf = OmegaConf.load(args.config_path)
  print(colorama.Fore.RED + str(args) + colorama.Style.RESET_ALL)

  np.random.seed(args.seed)
  Tensor.manual_seed(args.seed)

  logger = None # TODO: implement logger
  os.makedirs(f"runs/{args.n}", exist_ok=True) # until logger is implemented
  shutil.copy(args.config_path, f"runs/{args.n}/config.yaml")

  dummy_env = build_single_env(args.env_name, conf.basic_settings.image_size, seed=0)
  action_dim = dummy_env.action_space.n.item()
  del dummy_env

  world_model = build_world_model(conf, action_dim)
  agent = build_agent(conf, action_dim)

  replay_buffer = ReplayBuffer(
      obs_shape=(conf.basic_settings.image_size, conf.basic_settings.image_size, 3),
      num_envs=conf.joint_train_agent.num_envs,
      max_length=conf.joint_train_agent.buffer_max_length,
      warmup_length=conf.joint_train_agent.buffer_warmup,
  )

  if conf.joint_train_agent.use_demonstration:
    print(colorama.Fore.MAGENTA + f"loading demonstration trajectory from {args.trajectory_path}" + colorama.Style.RESET_ALL)
    replay_buffer.load_trajectory(path=args.trajectory_path)

  joint_train_world_model_agent(
    env_name=args.env_name, num_envs=conf.joint_train_agent.num_envs,
    max_steps=conf.joint_train_agent.sample_max_steps, image_size=conf.basic_settings.image_size,
    replay_buffer=replay_buffer, world_model=world_model, agent=agent,
    train_dynamics_every_steps=conf.joint_train_agent.train_dynamics_every_steps,
    train_agent_every_steps=conf.joint_train_agent.train_agent_every_steps,
    batch_size=conf.joint_train_agent.batch_size,
    demonstration_batch_size=conf.joint_train_agent.demonstration_batch_size if conf.joint_train_agent.use_demonstration else 0,
    batch_length=conf.joint_train_agent.batch_length,
    imagine_batch_size=conf.joint_train_agent.imagine_batch_size,
    imagine_demonstration_batch_size=conf.joint_train_agent.imagine_demonstration_batch_size if conf.joint_train_agent.use_demonstration else 0,
    imagine_context_length=conf.joint_train_agent.imagine_context_length,
    imagine_batch_length=conf.joint_train_agent.imagine_batch_length,
    save_every_steps=conf.joint_train_agent.save_every_steps,
    seed=args.seed, logger=logger
  )




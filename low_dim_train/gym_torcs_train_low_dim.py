"""
Implementation of DDPG - Deep Deterministic Policy Gradient - on gym-torcs.
with tensorflow.

ddpg paper:
    http://arxiv.org/pdf/1509.02971v2.pdf

Author: kenneth yu
"""
import tensorflow as tf
from gym_torcs.gym_torcs import TorcsEnv
from common.replay_buffer import preprocess_low_dim
import numpy as np
from common.common import env_step
import time


DDPG_CFG = tf.app.flags.FLAGS  # alias
DDPG_CFG.action_fields = ['steer', 'accel', 'brake']

from low_dim_train.train_agent_low_dim import  train

# torcs

DDPG_CFG.torcs_relaunch_freq = 3  #for memory leak bug.
DDPG_CFG.greedy_accel_noise_steps = 1*(10**5)

DDPG_CFG.policy_output_idx_steer = 0
DDPG_CFG.policy_output_idx_accel = 1
DDPG_CFG.policy_output_idx_brake = 2

# x is from BN.
def scale_sigmoid(x):
  return tf.nn.sigmoid(x * 3.3)  # when x==1, result is 0.964

# x is from BN.
def scale_tanh(x):
  return tf.nn.tanh(x * 2.0)  # when x==1, result is 0.964

## - [0]:steer [1]:accel [2]:brake --
DDPG_CFG.actor_output_bound_fns = [None for _ in range(len(DDPG_CFG.action_fields))]
DDPG_CFG.actor_output_bound_fns[DDPG_CFG.policy_output_idx_steer] = scale_tanh
DDPG_CFG.actor_output_bound_fns[DDPG_CFG.policy_output_idx_accel] = scale_sigmoid
DDPG_CFG.actor_output_bound_fns[DDPG_CFG.policy_output_idx_brake] = scale_sigmoid



DDPG_CFG.log_dir = 'train/gym_torcs_low_dim/tf_log/'
DDPG_CFG.checkpoint_dir = 'train/gym_torcs_low_dim/chk_pnt/'
DDPG_CFG.replay_buffer_file_path = 'train/gym_torcs_low_dim/replay_buffer/'

# global var for eval
max_avg_episode_reward = -1000

tf.logging.set_verbosity(tf.logging.INFO)

class torcs_env_wrapper(TorcsEnv):
    def __init__(self,*args, **kwargs):
        super(torcs_env_wrapper,self).__init__(*args,**kwargs)
        self.reset_count = 0

    def make_state(self,obs):
      state = np.hstack( #datas are already normalized in gym_torcs.
        (obs.angle, obs.track, obs.trackPos, obs.speedX, obs.speedY, obs.speedZ,
         obs.wheelSpinVel, obs.rpm))
      return state

    def reset(self, relaunch=False):
        obs = self._reset(relaunch or ((self.reset_count % DDPG_CFG.torcs_relaunch_freq) == 0) )
        self.reset_count += 1
        return self.make_state(obs)

    def step(self, action):
        obs, reward, term, _ = self._step(action)
        return self.make_state(obs), reward,term,_



def agent_action(policy_out, replay_buffer,env):
    ##add noise and bound
    stochastic_action=policy_output_to_stochastic_action(policy_out, env.action_space)

    ## excute a_t and store Transition.
    (state, reward, terminated) = env_step(env, stochastic_action)

    # replace transition with new one.
    transition = preprocess_low_dim(action=stochastic_action,
        reward=reward,
        terminated=terminated,
        state=state)
    ##even if terminated ,we still save next_state.
    replay_buffer.store(transition)
    return transition

def greedy_function(x, mu, theta, sigma):
        return theta * (mu - x) + sigma * np.random.randn()

epsilon=1
def policy_output_to_stochastic_action(output, action_space):
  global epsilon
  output = np.squeeze(output, axis=0)

  epsilon -= 1.0 / DDPG_CFG.greedy_accel_noise_steps

  greedy_noise=np.array( [max(epsilon, 0) * greedy_function(output[0], 0.0, 0.60, 0.30),  # steer
                  max(epsilon, 0) * greedy_function(output[1], 0.5, 1.00, 0.10),  # accel
                  max(epsilon, 0) * greedy_function(output[2], -0.1, 1.00, 0.05)]) # brake

  stochastic_action = greedy_noise + output
  bounded = np.clip(stochastic_action, action_space.low, action_space.high)
  return bounded


if __name__ == "__main__":
  tf.logging.info("@@@  start ddpg training gym_torcs @@@ start time:{}".format(time.ctime()))
  # Generate a Torcs environment
  env_train = torcs_env_wrapper(vision=False, throttle=True, gear_change=False,port=3101)
  train(env_train,agent_action,eval_mode=False)



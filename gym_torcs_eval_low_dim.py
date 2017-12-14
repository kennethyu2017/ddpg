"""
Implementation of DDPG - Deep Deterministic Policy Gradient - on gym-torcs.
with tensorflow.

ddpg paper:
    http://arxiv.org/pdf/1509.02971v2.pdf

Author: kenneth yu
"""
import time

import tensorflow as tf

from gym_torcs_train_low_dim import torcs_env_wrapper
from low_dim_train.train_agent_low_dim import train

DDPG_CFG = tf.app.flags.FLAGS  # alias
DDPG_CFG.log_dir = 'eval/gym_torcs_low_dim/tf_log/'
DDPG_CFG.checkpoint_dir = 'eval/gym_torcs_low_dim/chk_pnt/'
DDPG_CFG.eval_monitor_dir = 'eval/gym_torcs_low_dim/eval_monitor/'


tf.logging.set_verbosity(tf.logging.INFO)


if __name__ == "__main__":
  tf.logging.info("@@@  start ddpg evaluation gym_torcs @@@ start time:{}".format(time.ctime()))
  # Generate a Torcs environment
  env = torcs_env_wrapper(vision=True, throttle=True, gear_change=False)
  train(env,None,eval_mode=True)







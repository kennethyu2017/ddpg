"""
Implementation of DDPG - Deep Deterministic Policy Gradient - on gym-torcs.
with tensorflow.

ddpg paper:
    http://arxiv.org/pdf/1509.02971v2.pdf

Author: kenneth yu
"""
import tensorflow as tf
import numpy as np
from collections import deque,namedtuple
import copy
import os
import glob
import pickle as pkl
import time

DDPG_CFG = tf.app.flags.FLAGS  # alias

DDPG_CFG.replay_buffer_file_name = 'replay_buffer'
transition_fields = ['action', 'reward', 'terminated', 'next_state']
Transition = namedtuple('Transition', transition_fields)

def preprocess_low_dim(state, action=None, reward=None, terminated=None):
  transition = Transition(action=action,
                          reward=reward,
                          terminated=terminated,
                          next_state=state)
  return transition


class ReplayBuffer(object):
  def __init__(self, buffer_size, seed, save_segment_size=None, save_path=None):
    #The right side of the deque contains the most recent experiences
    self.buffer_size = buffer_size
    self.buffer = deque([], maxlen=buffer_size)
    if seed is not None:
      np.random.seed(seed)
    self.save=False
    if save_segment_size is not None:
      assert save_path is not None
      self.save = True
      self.save_segment_size = save_segment_size
      self.save_path = save_path
      self.save_data_cnt=0
      self.save_segment_cnt=0

  def store(self, transition):
    #deque can take care of max len.
    T = copy.deepcopy(transition)
    self.buffer.append(T)
    if self.save:
      self.save_data_cnt+=1
      if self.save_data_cnt >= self.save_segment_size:
        self.save_segment()
        self.save_data_cnt=0
    del transition

  def get_item(self,idx):
    return self.buffer[idx]

  @property
  def length(self):
    return len(self.buffer)

  @property
  def size(self):
    return self.buffer.__sizeof__()

  def sample_batch(self, batch_size):
    indices = np.random.permutation(self.length - 1)[:batch_size]
    state_batch, action_batch, reward_batch, next_state_batch, terminated_batch = [], [], [], [], []
    for idx in indices:
      trans_1 = self.buffer[idx]
      if trans_1.terminated is not True:
        # the trans_2 : (a_2, r_2, term_2, s_3)
        trans_2 = self.buffer[idx + 1]
        # we use the data (s_2, a_2, r_2, term_2, s_3)
        state_batch.append(trans_1.next_state)
        action_batch.append(trans_2.action)
        reward_batch.append(trans_2.reward)
        next_state_batch.append(trans_2.next_state)
        terminated_batch.append(trans_2.terminated)
      else:
        ##term_1 is true, so buffer[idx+1] is beginning of new episode,
        # we traverse back.
        if idx != 0:
          trans_0 = self.buffer[idx - 1]  # a_0, r_0, s_1, term_0 = self.buffer[idx - 1]
          if trans_0.terminated is True:  # give up
            continue
          # we use the data (s_1, a_1, r_1, term_1, s_2)
          # s_2 is not used to calc Q cause its terminated. but we still use
          # it to FF through mu_prime/Q_prime then Q*0. guarantee s_2 is accurate formatted.
          state_batch.append(trans_0.next_state)
          action_batch.append(trans_1.action)
          reward_batch.append(trans_1.reward)
          next_state_batch.append(trans_1.next_state)
          terminated_batch.append(trans_1.terminated)
        else:
          # head of buffer, we dont know the previous state , so give up.
          continue

    return (np.array(state_batch), np.array(action_batch), np.array(reward_batch), np.array(next_state_batch),np.array(terminated_batch))

  def save_segment(self):
    self.save_segment_cnt+=1

    data = []
    start = self.length - self.save_segment_size  #always save latest data of segment_size
    end = self.length

    for idx in range(start, end):
      data.append(self.buffer[idx])

    if not os.path.exists(self.save_path):
      os.makedirs(self.save_path)

    abs_file_name = os.path.abspath(os.path.join(self.save_path,
                            '_'.join([DDPG_CFG.replay_buffer_file_name,str(self.save_segment_cnt),time.ctime()])))

    with open(abs_file_name,'wb') as f:
      pkl.dump(data, f)


  def load(self, path):
    #load from file to buffer
    abs_file_pattern = os.path.abspath(os.path.join(path,
                            '_'.join([DDPG_CFG.replay_buffer_file_name,'*'])))
    buffer_files = glob.glob(abs_file_pattern)
    for f_name in buffer_files:
      with open(f_name,'rb') as f:
        data = pkl.load(f)
        self.buffer.extend(data)

  def clear(self):
    self.buffer.clear()


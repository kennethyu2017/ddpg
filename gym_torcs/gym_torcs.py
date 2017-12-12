import collections as col
import copy
import os
import time

import numpy as np
from gym import spaces
from gym.envs.registration import EnvSpec

import gym_torcs.snakeoil3_gym as snakeoil3

pi = 3.1416


def end():
  os.system('pkill torcs')


def obs_vision_to_image_rgb(obs_image_vec):
  image_vec = obs_image_vec
  rgb = []
  temp = []
  # convert size 64x64x3 = 12288 to 64x64=4096 2-D list
  # with rgb values grouped together.
  # Format similar to the observation in openai gym
  for i in range(0, 12286, 3):
    temp.append(image_vec[i])
    temp.append(image_vec[i + 1])
    temp.append(image_vec[i + 2])
    rgb.append(temp)
    temp = []
  return np.array(rgb, dtype=np.uint8)


class TorcsEnv:

  def __init__(self, vision=False, throttle=False, gear_change=False,port=3101):
    self.vision = vision
    self.throttle = throttle
    self.gear_change = gear_change
    self.port=port

    self.initial_run = True
    self.observation = None
    self.client = None
    self.time_step = 0
    self.last_u = None

    self.default_speed =200  #km/h

    self.low_progress_cnt = 0
    self.low_progress=0.05  # meters
    self.low_progress_term_steps=200 # term if no progress along track made after 200 steps.

    self.initial_reset = True

    os.system('pkill torcs')
    time.sleep(0.5)
    if self.vision is True:
      os.system('torcs  -nofuel -nodamage -nolaptime  -vision  &')
    else:
      os.system('torcs  -nofuel -nodamage -nolaptime -T &')
    time.sleep(0.5)
    os.system('sh /home/yuheng/PycharmProjects/rl/kenneth_ddpg/ddpg_add_low_dim_Dec07_for_pub/gym_torcs/autostart.sh')
    time.sleep(0.5)

    """
        # Modify here if you use multiple tracks in the environment
        self.client = snakeoil3.Client(p=self.port, vision=self.vision)  # Open new UDP in vtorcs
        self.client.MAX_STEPS = np.inf

        client = self.client
        client.get_servers_input()  # Get the initial input from torcs

        obs = client.S.d  # Get the current full-observation from torcs
        """
    if throttle is False:
      self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,))
    else:
      # steering, accel, brake.
      high = np.array([1., 1., 1.])
      low = np.array([-1., 0., 0.])
      self.action_space = spaces.Box(low=low, high=high)

    # vision case. ref below. #
    if vision is False:
      # ['speedX', 'speedY', 'speedZ', 'angle',
      #  'rpm',
      #  'track',
      #  'trackPos',
      #  'wheelSpinVel'] we only choose 9 params with 29 dims total:
      high = np.array([np.inf] * 29)
      low = np.array([-np.inf] * 29)
      self.observation_space = spaces.Box(low=low, high=high)
    else:
      high = np.array([np.inf] * 29)
      low = np.array([-np.inf] * 29)
      self.observation_space = spaces.Box(low=low, high=high)

  def _step(self, u):
    # print("Step")
    # convert thisAction to the actual torcs actionstr
    client = self.client

    this_action = self.agent_to_torcs(u)

    # Apply Action
    action_torcs = client.R.d

    # Steering
    action_torcs['steer'] = this_action['steer']  # in [-1, 1]

    #  Simple Autnmatic Throttle Control by Snakeoil
    if self.throttle is False:
      target_speed = self.default_speed
      if client.S.d['speedX'] < target_speed - (client.R.d['steer'] * 50):
        client.R.d['accel'] += .01
      else:
        client.R.d['accel'] -= .01

      if client.R.d['accel'] > 0.2:
        client.R.d['accel'] = 0.2

      if client.S.d['speedX'] < 10:
        client.R.d['accel'] += 1 / (client.S.d['speedX'] + .1)

      # Traction Control System
      if ((client.S.d['wheelSpinVel'][2] + client.S.d['wheelSpinVel'][3]) -
            (client.S.d['wheelSpinVel'][0] + client.S.d['wheelSpinVel'][1]) > 5):
        action_torcs['accel'] -= .2
    else:
      action_torcs['accel'] = this_action['accel']  # [0,1] #
      action_torcs['brake'] = this_action['brake']  # [0,1] #

    # Automatic Gear Change by Snakeoil
    if self.gear_change is True:
      action_torcs['gear'] = this_action['gear']
    else:
      #  Automatic Gear Change by Snakeoil is possible
      action_torcs['gear'] = 1
      if self.throttle:
            if client.S.d['speedX'] > 50:
                action_torcs['gear'] = 2
            if client.S.d['speedX'] > 80:
                action_torcs['gear'] = 3
            if client.S.d['speedX'] > 110:
                action_torcs['gear'] = 4
            if client.S.d['speedX'] > 140:
                action_torcs['gear'] = 5
            if client.S.d['speedX'] > 170:
                action_torcs['gear'] = 6
    # Save the privious full-obs from torcs for the reward calculation
    obs_pre = copy.deepcopy(client.S.d)

    # One-Step Dynamics Update #################################
    # Apply the Agent's action into torcs
    client.respond_to_server()
    # Get the response of TORCS
    client.get_servers_input()

    # Get the current full-observation from torcs
    obs = client.S.d

    # Make an obsevation from a raw observation vector from TORCS
    ## Note: this state is after the race_server execute the actions.
    ## so it should be s_t+1.
    self.observation = self.make_observaton(obs)

    # Reward setting Here #######################################
    # direction-dependent positive reward
    track = np.array(obs['track'])
    trackPos = np.array(obs['trackPos'])
    sp = np.array(obs['speedX'])  #un-normalized velocity
    damage = np.array(obs['damage'])
    rpm = np.array(obs['rpm'])
    dist=obs['distFromStart']  # dist from start line along the track line.


    progress = sp * np.cos(obs['angle'])

    reward = progress
    episode_terminate = False

    # collision detection. not term.
    if obs['damage'] - obs_pre['damage'] > 0:
      reward = -1

    # Termination judgement #########################
    if abs(obs['trackPos']) > 1:  # Episode is terminated if the car is out of track
      reward = - 1
      episode_terminate = True
      client.R.d['meta'] = True

    # term if no progress along the track after limit consecutive steps:
    if obs['distFromStart'] - obs_pre['distFromStart'] > self.low_progress:
      self.low_progress_cnt = 0 #clear
    else:
      self.low_progress_cnt += 1

    # allow some time to tolerate no progress and maybe the robot can adjust to find speed up.
    if self.low_progress_cnt == self.low_progress_term_steps:
      reward = - 1
      self.low_progress_cnt = 0 #clear
      episode_terminate = True
      client.R.d['meta'] = True


    if np.cos(obs['angle']) < 0:  # Episode is terminated if the agent runs backward
      episode_terminate = True
      client.R.d['meta'] = True

    if client.R.d['meta'] is True:  # Send a reset signal
      self.initial_run = False
      client.respond_to_server()

    self.time_step += 1

    return self.get_obs(), reward, client.R.d['meta'], {}

  def _reset(self, relaunch=False):
    # print("Reset")

    self.time_step = 0

    if self.initial_reset is not True:
      self.client.R.d['meta'] = True
      self.client.respond_to_server()

      ## TENTATIVE. Restarting TORCS every episode suffers the memory leak bug!
      if relaunch is True:
        self.reset_torcs()
        print("### TORCS is RELAUNCHED ###")

    # Modify here if you use multiple tracks in the environment
    self.client = snakeoil3.Client(p=self.port, vision=self.vision)  # Open new UDP in vtorcs
    self.client.MAX_STEPS = np.inf

    client = self.client
    client.get_servers_input()  # Get the initial input from torcs

    obs = client.S.d  # Get the current full-observation from torcs
    self.observation = self.make_observaton(obs)

    self.last_u = None

    self.initial_reset = False
    return self.get_obs()

  def get_obs(self):
    return self.observation

  def reset_torcs(self):
    # print("relaunch torcs")
    os.system('pkill torcs')
    time.sleep(0.5)
    if self.vision is True:
      os.system('torcs -nofuel -nodamage -nolaptime -vision &')
    else:
      os.system('torcs -nofuel -nodamage -nolaptime -T &')
    time.sleep(0.5)
    # kenneth
    os.system('sh /home/yuheng/PycharmProjects/rl/kenneth_ddpg/ddpg_add_low_dim_Dec07_for_pub/gym_torcs/autostart.sh')
    time.sleep(0.5)

  def agent_to_torcs(self, u):
    torcs_action = {'steer': u[0]}

    if self.throttle is True:  # throttle action is enabled
      torcs_action.update({'accel': u[1]})
      torcs_action.update({'brake': u[2]})

    if self.gear_change is True:  # gear change action is enabled
      torcs_action.update({'gear': int(u[3])})

    return torcs_action

  def make_observaton(self, raw_obs):
    if self.vision is False:
      names = ['focus',
               'speedX', 'speedY', 'speedZ', 'angle', 'damage',
               'opponents',
               'rpm',
               'track',
               'trackPos',
               'wheelSpinVel']
      Observation = col.namedtuple('Observaion', names)

      #TODO kenneth . normalize to [-1,+1]
      return Observation(focus=np.array(raw_obs['focus'], dtype=np.float32) / 200.,
                         speedX=np.array(raw_obs['speedX'], dtype=np.float32) / self.default_speed,
                         speedY=np.array(raw_obs['speedY'], dtype=np.float32) / self.default_speed,
                         speedZ=np.array(raw_obs['speedZ'], dtype=np.float32) / self.default_speed,
                         angle=np.array(raw_obs['angle'], dtype=np.float32) / pi,
                         damage=np.array(raw_obs['damage'], dtype=np.float32),
                         opponents=np.array(raw_obs['opponents'], dtype=np.float32) / 200.,
                         rpm=np.array(raw_obs['rpm'], dtype=np.float32)/10000,
                         track=np.array(raw_obs['track'], dtype=np.float32) / 200.,
                         trackPos=np.array(raw_obs['trackPos'], dtype=np.float32),
                         wheelSpinVel=np.array(raw_obs['wheelSpinVel'], dtype=np.float32)/1000. )
    else:
      names = ['focus',
               'speedX', 'speedY', 'speedZ',
               'angle',
               'damage',
               'opponents',
               'rpm',
               'track',
               'trackPos',
               'wheelSpinVel',
               'img']
      Observation = col.namedtuple('Observaion', names)

      # Get RGB from observation
      image_rgb = obs_vision_to_image_rgb(raw_obs[names[-1]])

      return Observation(focus=np.array(raw_obs['focus'], dtype=np.float32) / 200.,
                         speedX=np.array(raw_obs['speedX'], dtype=np.float32) / self.default_speed,
                         speedY=np.array(raw_obs['speedY'], dtype=np.float32) / self.default_speed,
                         speedZ=np.array(raw_obs['speedZ'], dtype=np.float32) / self.default_speed,
                         angle=np.array(raw_obs['angle'], dtype=np.float32) / pi,
                         damage=np.array(raw_obs['damage'], dtype=np.float32),
                         opponents=np.array(raw_obs['opponents'], dtype=np.float32) / 200.,
                         rpm=np.array(raw_obs['rpm'], dtype=np.float32)/10000,
                         track=np.array(raw_obs['track'], dtype=np.float32) / 200.,
                         trackPos=np.array(raw_obs['trackPos'], dtype=np.float32) / 1.,
                         wheelSpinVel=np.array(raw_obs['wheelSpinVel'], dtype=np.float32)/1000,
                         img=image_rgb)


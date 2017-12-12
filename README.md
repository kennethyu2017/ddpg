# ddpg
Implementation of DDPG - Deep Deterministic Policy Gradient - on gym-torcs.
with tensorflow.

ddpg paper:
    http://arxiv.org/pdf/1509.02971v2.pdf

Author: kenneth yu

installation dependencies:
  1. tensorflow r1.4
  2. gym_torcs: https://github.com/ugo-nama-kun/gym_torcs
  
how to run:
  1. training mode: 
   ```shell
    python3 gym_torcs_train_low_dim.py
   ```
  2. evaluate mode:
  ```shell
    python3 gym_torcs_eval_low_dim.py
  ```

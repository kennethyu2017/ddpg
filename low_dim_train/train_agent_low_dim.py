"""
Implementation of DDPG - Deep Deterministic Policy Gradient - on gym-torcs.
with tensorflow.

ddpg paper:
    http://arxiv.org/pdf/1509.02971v2.pdf

Author: kenneth yu
"""
import os
import numpy as np
import tensorflow as tf
from low_dim_train.actor_low_dim import Actor
from low_dim_train.critic_low_dim import Critic
from tensorflow.contrib.layers import variance_scaling_initializer,batch_norm,l2_regularizer
from tensorflow.python.framework.dtypes import string
from common.replay_buffer import ReplayBuffer, preprocess_low_dim
from common.common import env_step,policy_output_to_deterministic_action


DDPG_CFG = tf.app.flags.FLAGS  # alias
DDPG_CFG.random_seed = 187
np.random.seed(DDPG_CFG.random_seed)

DDPG_CFG.train_from_replay_buffer_set_only = False
DDPG_CFG.load_replay_buffer_set = False

## hyper-P
DDPG_CFG.actor_learning_rate = 1e-3
DDPG_CFG.critic_learning_rate = 1e-4
DDPG_CFG.critic_reg_ratio = 1e-2
DDPG_CFG.tau = 0.001
DDPG_CFG.gamma = 0.99
DDPG_CFG.num_training_steps = 25*(10**5)  # 2.5M steps total
DDPG_CFG.eval_freq = 3*10000
DDPG_CFG.num_eval_steps = 1000  # eval steps during training
DDPG_CFG.eval_steps_after_training=2000
DDPG_CFG.batch_size = 64
DDPG_CFG.replay_buff_size = 10**6  # 1M
DDPG_CFG.replay_buff_save_segment_size = 30*3000  # every 180,000 Transition data.
DDPG_CFG.log_summary_keys = 'log_summaries'

# === actor net arch. shared by online and target ==
is_training=tf.placeholder(tf.bool, shape=(), name='is_training')

DDPG_CFG.online_policy_net_var_scope = 'online_policy'
DDPG_CFG.target_policy_net_var_scope = 'target_policy'

# -- 1 input norm layers --
DDPG_CFG.actor_input_normalizer = batch_norm
DDPG_CFG.actor_input_norm_params =  {'is_training':is_training,
                                       'data_format':'NHWC',
                                       'updates_collections':None,
                                        'scale':False,  # not gamma. let next fc layer to scale.
                                        'center':True  # beta.
                                        }
# -- 2 fc layers --
DDPG_CFG.actor_n_fc_units = [400, 300]
DDPG_CFG.actor_fc_activations = [tf.nn.elu] * 2
DDPG_CFG.actor_fc_initializers = [variance_scaling_initializer()] * 2
DDPG_CFG.actor_fc_regularizers = [None] * 2
DDPG_CFG.actor_fc_normalizers = [batch_norm] * 2
DDPG_CFG.actor_fc_norm_params =  [{'is_training':is_training,
                                       'data_format':'NHWC',
                                       'updates_collections':None,
                                       'scale':False,
                                       'center':True
                                         }] *2

# -- 1 output layer --
#TODO try actor no BN.use l2 reg on weights only.
DDPG_CFG.actor_output_layer_normalizers = batch_norm
DDPG_CFG.actor_output_layer_norm_params = {'is_training':is_training,
                                       'data_format':'NHWC',
                                       'updates_collections':None,
                                       'scale':False,
                                       'center':False}
DDPG_CFG.actor_output_layer_initializer=tf.random_uniform_initializer(-3e-3,3e-3)

# === critic net arch. shared by online and target ==
DDPG_CFG.online_q_net_var_scope = 'online_q'
DDPG_CFG.target_q_net_var_scope = 'target_q'
DDPG_CFG.critic_summary_keys = 'critic_summaries'

#-- 1 input norm layers --
DDPG_CFG.critic_input_normalizer = batch_norm
DDPG_CFG.critic_input_norm_params =  {'is_training':is_training,
                                       'data_format':'NHWC',
                                       'updates_collections':None,
                                       'scale': False,
                                       'center': True
                                      }

# -- 2 fc layer --
DDPG_CFG.include_action_fc_layer = 2  # in this layer we include action inputs. conting from fc-1 as 1.
DDPG_CFG.critic_n_fc_units = [400, 300]
DDPG_CFG.critic_fc_activations = [tf.nn.elu] * 2
DDPG_CFG.critic_fc_initializers = [variance_scaling_initializer()] * 2
DDPG_CFG.critic_fc_regularizers = [l2_regularizer(scale=DDPG_CFG.critic_reg_ratio,scope=DDPG_CFG.online_q_net_var_scope)] * 2
DDPG_CFG.critic_fc_normalizers = [batch_norm, None]  # 2nd fc including action input and no BN but has bias.
DDPG_CFG.critic_fc_norm_params =  [{'is_training':is_training,
                                       'data_format':'NHWC',
                                       'updates_collections':None,
                                       'scale': False,
                                       'center': True
                                    }, None]

# -- 1 output layer --
DDPG_CFG.critic_output_layer_initializer = tf.random_uniform_initializer(-3e-3, 3e-3)

tf.logging.set_verbosity(tf.logging.INFO)

def build_ddpg_graph(actor, critic, reward_inputs, terminated_inputs, global_step_tensor):
  y_i = reward_inputs + (1.0 - terminated_inputs) * DDPG_CFG.gamma * critic.target_q_outputs_tensor
  #list of reg scalar Tensors
  q_reg_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES,scope=DDPG_CFG.online_q_net_var_scope)
  q_loss = tf.add_n([tf.losses.mean_squared_error(labels=y_i, predictions=critic.online_q_outputs_tensor)] + q_reg_loss,
                    name='q_loss')
  #list of reg scalar Tensors.
  policy_reg_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, scope=DDPG_CFG.online_policy_net_var_scope)
  policy_loss =tf.add_n([ -1.0 * tf.reduce_mean(critic.online_q_outputs_tensor)] + policy_reg_loss,
                        name='policy_loss')

  # -- optimize actor online network --
  actor_g_and_v, actor_compute_grads_op = actor.compute_online_policy_net_gradients(policy_loss=policy_loss)
  actor_apply_grads_op = actor.apply_online_policy_net_gradients(grads_and_vars=actor_g_and_v)
  train_online_policy_op = actor_apply_grads_op

  # -- optimize critic online network -
  critic_g_and_v, critic_compute_grads_op = critic.compute_online_q_net_gradients(q_loss=q_loss)
  critic_apply_grads_op = critic.apply_online_q_net_gradients(grads_and_vars=critic_g_and_v)
  train_online_q_op = critic_apply_grads_op

  actor_update_target_op = actor.soft_update_online_to_target()
  critic_update_target_op = critic.soft_update_online_to_target()

  ##increment global step <- soft update ops
  with tf.control_dependencies([actor_update_target_op, critic_update_target_op]):
    update_target_op = tf.assign_add(global_step_tensor, 1).op  # increment global step

  # after params initial, copy online -> target
  actor_init_target_op = actor.copy_online_to_target()
  critic_init_target_op = critic.copy_online_to_target()

  copy_online_to_target_op = tf.group(actor_init_target_op, critic_init_target_op)

  # model saver
  saver = tf.train.Saver(keep_checkpoint_every_n_hours=0.5, max_to_keep=5)

  return (copy_online_to_target_op, train_online_policy_op, train_online_q_op, update_target_op, saver)


def train(train_env, agent_action_fn, eval_mode=False):
  action_space = train_env.action_space
  obs_space = train_env.observation_space

  ######### instantiate actor,critic, replay buffer, uo-process#########
  ## feed online with state. feed target with next_state.
  online_state_inputs = tf.placeholder(tf.float32,
                                       shape=(None, obs_space.shape[0]),
                                       name="online_state_inputs")

  target_state_inputs = tf.placeholder(tf.float32,
                                       shape=online_state_inputs.shape,
                                       name="target_state_inputs")

  ## inputs to q_net for training q.
  online_action_inputs_training_q = tf.placeholder(tf.float32,
                                                   shape=(None, action_space.shape[0]),
                                                   name='online_action_batch_inputs'
                                                   )
  # condition bool scalar to switch action inputs to online q.
  # feed True: training q.
  # feed False: training policy.
  cond_training_q = tf.placeholder(tf.bool, shape=[], name='cond_training_q')

  terminated_inputs = tf.placeholder(tf.float32, shape=(None), name='terminated_inputs')
  reward_inputs = tf.placeholder(tf.float32, shape=(None), name='rewards_inputs')

  # for summary text
  summary_text_tensor = tf.convert_to_tensor(str('summary_text'), preferred_dtype=string)
  tf.summary.text(name='summary_text', tensor=summary_text_tensor, collections=[DDPG_CFG.log_summary_keys])

  ##instantiate actor, critic.
  actor = Actor(action_dim=action_space.shape[0],
                online_state_inputs=online_state_inputs,
                target_state_inputs=target_state_inputs,
                input_normalizer=DDPG_CFG.actor_input_normalizer,
                input_norm_params=DDPG_CFG.actor_input_norm_params,
                n_fc_units=DDPG_CFG.actor_n_fc_units,
                fc_activations=DDPG_CFG.actor_fc_activations,
                fc_initializers=DDPG_CFG.actor_fc_initializers,
                fc_normalizers=DDPG_CFG.actor_fc_normalizers,
                fc_norm_params=DDPG_CFG.actor_fc_norm_params,
                fc_regularizers=DDPG_CFG.actor_fc_regularizers,
                output_layer_initializer=DDPG_CFG.actor_output_layer_initializer,
                output_layer_regularizer=None,
                output_normalizers=DDPG_CFG.actor_output_layer_normalizers,
                output_norm_params=DDPG_CFG.actor_output_layer_norm_params,
                output_bound_fns=DDPG_CFG.actor_output_bound_fns,
                learning_rate=DDPG_CFG.actor_learning_rate,
                is_training=is_training)


  critic = Critic(online_state_inputs=online_state_inputs,
                  target_state_inputs=target_state_inputs,
                  input_normalizer=DDPG_CFG.critic_input_normalizer,
                  input_norm_params=DDPG_CFG.critic_input_norm_params,
                  online_action_inputs_training_q=online_action_inputs_training_q,
                  online_action_inputs_training_policy=actor.online_action_outputs_tensor,
                  cond_training_q=cond_training_q,
                  target_action_inputs=actor.target_action_outputs_tensor,
                  n_fc_units=DDPG_CFG.critic_n_fc_units,
                  fc_activations=DDPG_CFG.critic_fc_activations,
                  fc_initializers=DDPG_CFG.critic_fc_initializers,
                  fc_normalizers=DDPG_CFG.critic_fc_normalizers,
                  fc_norm_params=DDPG_CFG.critic_fc_norm_params,
                  fc_regularizers=DDPG_CFG.critic_fc_regularizers,
                  output_layer_initializer=DDPG_CFG.critic_output_layer_initializer,
                  output_layer_regularizer = None,
                  learning_rate=DDPG_CFG.critic_learning_rate)

  ## track updates.
  global_step_tensor = tf.train.create_global_step()

  ## build whole graph
  copy_online_to_target_op, train_online_policy_op, train_online_q_op, update_target_op, saver \
    = build_ddpg_graph(actor, critic, reward_inputs, terminated_inputs, global_step_tensor)

  #we save the replay buffer data to files.
  replay_buffer = ReplayBuffer(buffer_size=DDPG_CFG.replay_buff_size,
                               save_segment_size= DDPG_CFG.replay_buff_save_segment_size,
                               save_path=DDPG_CFG.replay_buffer_file_path,
                               seed=DDPG_CFG.random_seed
                               )
  if DDPG_CFG.load_replay_buffer_set:
    replay_buffer.load(DDPG_CFG.replay_buffer_file_path)

  sess = tf.Session(graph=tf.get_default_graph())
  summary_writer = tf.summary.FileWriter(logdir=os.path.join(DDPG_CFG.log_dir, "train"),
                                         graph=sess.graph)
  log_summary_op = tf.summary.merge_all(key=DDPG_CFG.log_summary_keys)

  sess.run(fetches=[tf.global_variables_initializer()])

  #copy init params from online to target
  sess.run(fetches=[copy_online_to_target_op])

  # Load a previous checkpoint if it exists
  latest_checkpoint = tf.train.latest_checkpoint(DDPG_CFG.checkpoint_dir)
  if latest_checkpoint:
    tf.logging.info("==== Loading model checkpoint: {}".format(latest_checkpoint))
    saver.restore(sess, latest_checkpoint)

  ####### start training #########
  obs = train_env.reset()
  transition = preprocess_low_dim(obs)

  n_episodes = 1

  if not eval_mode:
    for step in range(1, DDPG_CFG.num_training_steps):
      #replace with new transition
      policy_out=sess.run(fetches=[actor.online_action_outputs_tensor],
               feed_dict={online_state_inputs: transition.next_state[np.newaxis, :], is_training: False})[0]
      transition=agent_action_fn(policy_out,
                                 replay_buffer,
                                 train_env)
      if step % 200 ==0 :
        tf.logging.info(' +++++++++++++++++++ global_step:{} action:{}'
                        '  reward:{} term:{}'.format(step, transition.action,transition.reward,
                                                     transition.terminated))
      if step < 10 :
        #feed some transitions in buffer.
        continue
      ## ++++ sample mini-batch and train.++++
      state_batch, action_batch, reward_batch, next_state_batch, terminated_batch = \
       replay_buffer.sample_batch(DDPG_CFG.batch_size)

      # ---- 1. train policy.-----------
      sess.run(fetches=[train_online_policy_op],
               feed_dict={online_state_inputs: state_batch,
                          cond_training_q: False,
                          online_action_inputs_training_q: action_batch,  # feed but not used.
                          is_training: True
                          })

      # ---- 2. train q. --------------
      sess.run(fetches=[train_online_q_op],
                feed_dict={ online_state_inputs: state_batch,
                            cond_training_q: True,
                            online_action_inputs_training_q: action_batch,
                            target_state_inputs: next_state_batch,
                            reward_inputs: reward_batch,
                            terminated_inputs: terminated_batch,
                            is_training: True})

      # ----- 3. update target ---------
      sess.run(fetches=[update_target_op], feed_dict=None)

      # do evaluation after eval_freq steps:
      if step % DDPG_CFG.eval_freq == 0: ##and step > DDPG_CFG.eval_freq:
        evaluate(env=train_env,
                 num_eval_steps=DDPG_CFG.num_eval_steps,
                 preprocess_fn=preprocess_low_dim,
                 estimate_fn=lambda state: sess.run(fetches=[actor.online_action_outputs_tensor],
                                                      feed_dict={online_state_inputs:state,
                                                      is_training:False} ),
                 summary_writer=summary_writer,
                 saver=saver, sess=sess, global_step=step,
                 log_summary_op=log_summary_op,summary_text_tensor=summary_text_tensor)

      if transition.terminated:
        transition = preprocess_low_dim(train_env.reset())
        n_episodes +=1
        continue  # begin new episode

  else: #eval mode
    evaluate(env=train_env,
           num_eval_steps=DDPG_CFG.eval_steps_after_training,
           preprocess_fn=preprocess_low_dim,
           estimate_fn=lambda state: sess.run(fetches=[actor.online_action_outputs_tensor],
                                              feed_dict={online_state_inputs: state,
                                                         is_training: False}),
           summary_writer=summary_writer,
           saver=None, sess=sess, global_step=0,
           log_summary_op=log_summary_op, summary_text_tensor=summary_text_tensor)

  sess.close()
  train_env.close()

def evaluate(env, num_eval_steps, preprocess_fn, estimate_fn,
             summary_writer, saver, sess,global_step,log_summary_op,summary_text_tensor):
  total_reward = 0
  episode_reward = 0
  max_episode_reward = 0
  n_episodes = 0
  n_rewards = 0
  terminated = False
  transition = preprocess_fn(state=env.reset())

  tf.logging.info(' ####### start evaluate @ global step:{}##  '.format(global_step))

  for estep in range(1,num_eval_steps):
    policy_out = estimate_fn(transition.next_state[np.newaxis,:])
    action = policy_output_to_deterministic_action(policy_out,env.action_space)
    (state, reward, terminated) = env_step(env, action)

    # we only need state to generate policy.
    transition = preprocess_fn(state)

    # record every reward
    total_reward += reward
    episode_reward += reward

    if reward != 0:
      n_rewards += 1  # can represent effective(not still) steps in episode

    if terminated:
      n_episodes += 1
      if episode_reward > max_episode_reward:
        max_episode_reward = episode_reward
        episode_reward = 0
      # relaunch
      # only save state.
      transition = preprocess_fn(env.reset())

  # -- end for estep ---
  avg_episode_reward = total_reward / max(1, n_episodes)
  avg_episode_steps = n_rewards / max(1, n_episodes)

  #we  save model only during training.
  saved_name='eval_only_not_save_model'
  if saver is not None:
    saved_name = save_model(saver, sess, global_step)
  write_summary(summary_writer, global_step, avg_episode_reward, max_episode_reward,
                avg_episode_steps,saved_name,sess,log_summary_op,
                summary_text_tensor)

def write_summary(writer, global_step, avg_episode_reward, max_episode_reward,
                  avg_episode_steps,saved_name,sess,log_summary_op,
                  summary_text_tensor):
  eval_summary = tf.Summary()  # proto buffer
  eval_summary.value.add(node_name='avg_episode_reward',simple_value=avg_episode_reward, tag="train_eval/avg_episode_reward")
  eval_summary.value.add(node_name='max_episode_reward', simple_value=max_episode_reward, tag="train_eval/max_episode_reward")
  eval_summary.value.add(node_name='avg_episode_steps', simple_value=avg_episode_steps, tag="train_eval/avg_episode_steps")
  writer.add_summary(summary=eval_summary, global_step=global_step)

  #saved model name
  log_info = 'eval result : global_step:{}    avg_episode_reward:{} \
              max_episode_reward:{}   avg_episode_steps:{}  \n saved_file: {} '.format(global_step,
                                                                                            avg_episode_reward,
                                                                                            max_episode_reward,
                                                                                            avg_episode_steps,
                                                                                            saved_name)
  tf.logging.info(log_info)
  log_summary=sess.run(fetches=[log_summary_op],
                       feed_dict={summary_text_tensor:log_info})
  writer.add_summary(summary=log_summary[0], global_step=global_step)
  writer.flush()


def save_model(saver, sess,global_step):
  # save model. will save both online and target networks.
  return saver.save(sess=sess, save_path=DDPG_CFG.checkpoint_dir, global_step=global_step)


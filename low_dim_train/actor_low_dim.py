"""
Implementation of DDPG - Deep Deterministic Policy Gradient - on gym-torcs.
with tensorflow.

ddpg paper:
    http://arxiv.org/pdf/1509.02971v2.pdf

Author: kenneth yu
"""
import tensorflow as tf
from tensorflow.contrib.layers import fully_connected,batch_norm

from common.common import soft_update_online_to_target, copy_online_to_target

DDPG_CFG = tf.app.flags.FLAGS  # alias


class Actor(object):
  def __init__(self, action_dim,
               online_state_inputs, target_state_inputs,input_normalizer, input_norm_params,
               n_fc_units, fc_activations, fc_initializers,
               fc_normalizers, fc_norm_params, fc_regularizers,
               output_layer_initializer, output_layer_regularizer,
               output_normalizers, output_norm_params,output_bound_fns,
               learning_rate, is_training):
    self.a_dim = action_dim
    self.learning_rate = learning_rate

    self.online_state_inputs = online_state_inputs
    self.target_state_inputs = target_state_inputs
    self.input_normalizer = input_normalizer
    self.input_norm_params=input_norm_params

    self.n_fc_units = n_fc_units
    self.fc_activations = fc_activations
    self.fc_initializers = fc_initializers
    self.fc_normalizers = fc_normalizers
    self.fc_norm_params = fc_norm_params
    self.fc_regularizers =fc_regularizers

    self.output_layer_initializer = output_layer_initializer
    self.output_bound_fns = output_bound_fns  # bound each action dim.
    self.output_layer_regularizer = output_layer_regularizer
    self.output_normalizers = output_normalizers
    self.output_norm_params = output_norm_params
    self.is_training=is_training

    self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)  # use beta1, beta2 default.

    # online policy
    self._online_action_outputs = self.create_policy_net(scope=DDPG_CFG.online_policy_net_var_scope,
                                                        state_inputs=self.online_state_inputs,
                                                        trainable=True)

    self.online_policy_net_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                                    scope=DDPG_CFG.online_policy_net_var_scope)
    self.online_policy_net_vars_by_name = {var.name.strip(DDPG_CFG.online_policy_net_var_scope):var
                                           for var in self.online_policy_net_vars}

    # target policy
    # target net is untrainable.
    self._target_action_outputs = self.create_policy_net(scope=DDPG_CFG.target_policy_net_var_scope,
                                                        state_inputs=self.target_state_inputs,
                                                        trainable=False)
    self.target_policy_net_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                                    scope=DDPG_CFG.target_policy_net_var_scope)

    self.target_policy_net_vars_by_name = {var.name.strip(DDPG_CFG.target_policy_net_var_scope):var
                                           for var in self.target_policy_net_vars}

  def create_policy_net(self, state_inputs, scope, trainable):
    with tf.variable_scope(scope):
      #input norm layer
      prev_layer = self.input_normalizer(state_inputs, **self.input_norm_params)

      ##fc layers
      for n_unit, activation, initializer, normalizer, norm_param, regularizer in zip(
          self.n_fc_units, self.fc_activations, self.fc_initializers,
        self.fc_normalizers,self.fc_norm_params, self.fc_regularizers):
        prev_layer = fully_connected(prev_layer, num_outputs=n_unit, activation_fn=activation,
                                     weights_initializer=initializer,
                                     weights_regularizer=regularizer,
                                     normalizer_fn=normalizer,
                                     normalizer_params=norm_param,
                                     biases_initializer=None, #skip bias when use norm.
                                     trainable=trainable)

      ##output layer
      output_layer = fully_connected(prev_layer, num_outputs=self.a_dim, activation_fn=None,
                                     weights_initializer=self.output_layer_initializer,
                                     weights_regularizer=self.output_layer_regularizer,
                                     normalizer_fn=self.output_normalizers,
                                     normalizer_params=self.output_norm_params,
                                     biases_initializer=None, # to skip bias
                                     trainable=trainable)
      ## bound and scale each action dim
      action_unpacked = tf.unstack(output_layer, axis=1)
      action_bounded = []
      for i in range(self.a_dim):
        action_bounded.append(self.output_bound_fns[i](action_unpacked[i]))

      action_outputs = tf.stack(action_bounded, axis=1)

    return action_outputs

  # of online net
  @property
  def online_action_outputs_tensor(self):
    return self._online_action_outputs

  # of target net
  @property
  def target_action_outputs_tensor(self):
    return self._target_action_outputs

  def compute_online_policy_net_gradients(self, policy_loss):
    grads_and_vars = self.optimizer.compute_gradients(
      policy_loss,var_list=self.online_policy_net_vars)
    grads = [g for (g, _) in grads_and_vars if g is not None]
    compute_op = tf.group(*grads)

    return (grads_and_vars, compute_op)

  def apply_online_policy_net_gradients(self, grads_and_vars):
    vars_with_grad = [v for g, v in grads_and_vars if g is not None]
    if not vars_with_grad:
      raise ValueError(
        "$$ ddpg $$ policy net $$ No gradients provided for any variable, check your graph for ops"
        " that do not support gradients,variables %s." %
        ([str(v) for _, v in grads_and_vars]))
    return self.optimizer.apply_gradients(grads_and_vars)

  def soft_update_online_to_target(self):
    return soft_update_online_to_target(self.online_policy_net_vars_by_name,
                                        self.target_policy_net_vars_by_name)

  def copy_online_to_target(self):
    return copy_online_to_target(self.online_policy_net_vars_by_name,
                                 self.target_policy_net_vars_by_name)

"""
Implementation of DDPG - Deep Deterministic Policy Gradient - on gym-torcs.
with tensorflow.

ddpg paper:
    http://arxiv.org/pdf/1509.02971v2.pdf

Author: kenneth yu
"""
import tensorflow as tf
from tensorflow.contrib.layers import fully_connected
from tensorflow.python.layers.core import dense
from tensorflow.python.ops.control_flow_ops import switch,merge

from common.common import soft_update_online_to_target, copy_online_to_target

DDPG_CFG = tf.app.flags.FLAGS  # alias

class Critic(object):

  def __init__(self,
               online_state_inputs, target_state_inputs,input_normalizer,input_norm_params,
               online_action_inputs_training_q,
               online_action_inputs_training_policy,
               cond_training_q,
               target_action_inputs,
               n_fc_units, fc_activations, fc_initializers,
               fc_normalizers,fc_norm_params,fc_regularizers,
               output_layer_initializer,output_layer_regularizer,learning_rate):

    # TODO check gym-trocs state dims.
    self.online_state_inputs = online_state_inputs
    self.target_state_inputs = target_state_inputs
    self.input_normalizer = input_normalizer
    self.input_norm_params = input_norm_params
    # action to be included in fc-INCLUDE_ACTION_FC_LAYER layer.
    self.online_action_inputs_training_q = online_action_inputs_training_q
    self.online_action_inputs_training_policy = online_action_inputs_training_policy
    self.cond_training_q = cond_training_q

    self.target_action_inputs = target_action_inputs

    self.learning_rate = learning_rate

    self.n_fc_units = n_fc_units
    self.fc_activations = fc_activations
    self.fc_initializers = fc_initializers
    self.fc_normalizers = fc_normalizers
    self.fc_norm_params = fc_norm_params
    self.fc_regularizers = fc_regularizers

    self.output_layer_initializer = output_layer_initializer
    self.output_layer_regularizer = output_layer_regularizer

    self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)

    # online q net
    self.online_q_outputs = self.create_q_net(state_inputs=self.online_state_inputs,
                                              action_inputs_training_q=self.online_action_inputs_training_q,
                                              action_inputs_training_policy=self.online_action_inputs_training_policy,
                                              cond_training_q=self.cond_training_q,
                                              scope=DDPG_CFG.online_q_net_var_scope,
                                              trainable=True)

    self.online_q_net_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                               scope=DDPG_CFG.online_q_net_var_scope)
    self.online_q_net_vars_by_name = {var.name.strip(DDPG_CFG.online_q_net_var_scope):var
                                      for var in self.online_q_net_vars}

    # target q net. Untrainable ===
    ## we only need input actions for training q
    self.target_q_outputs = self.create_q_net(state_inputs=self.target_state_inputs,
                                         action_inputs_training_q=self.target_action_inputs,
                                         action_inputs_training_policy=None,
                                         cond_training_q=None,
                                         scope=DDPG_CFG.target_q_net_var_scope,
                                         trainable=False)

    self.target_q_net_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                               scope=DDPG_CFG.target_q_net_var_scope)

    self.target_q_net_vars_by_name = {var.name.strip(DDPG_CFG.target_q_net_var_scope):var
                                      for var in self.target_q_net_vars}

  ##TODO add one params to control no need init target. or use zeros_initializer?
  ### for online q: we need action input for training q and training policy both.cause its involved in training q and
  ## training policy.
  #   for target q: we need only the action input for training q  cause its involved only in training q.
  def create_q_net(self, state_inputs,  # NHWC format.
                   action_inputs_training_q,
                   scope, trainable,
                   action_inputs_training_policy=None,  # None for target net.
                   cond_training_q=None  # bool to control switch. can be None for target net.
                   ):
    with tf.variable_scope(scope):
      #input norm layer
      prev_layer=self.input_normalizer(state_inputs,**self.input_norm_params)

      ##fc layers
      l = 1  # start from fc-1 as 1
      for n_unit, activation, initializer, normalizer, norm_param,regularizer in zip(
              self.n_fc_units, self.fc_activations, self.fc_initializers,
              self.fc_normalizers, self.fc_norm_params, self.fc_regularizers):
        # include action_inputs
        if l == DDPG_CFG.include_action_fc_layer:
          if action_inputs_training_policy is None:  # target net
            actions = action_inputs_training_q
          else:  # add logic for selecting online net action inputs
            # switch return :(output_false, output_true)
            (_, sw_action_training_q) = switch(data=action_inputs_training_q,
                                                                   pred=cond_training_q,
                                                                   name='switch_actions_training_q')
            (sw_action_training_policy, _) = switch(data=action_inputs_training_policy,
                                                                        pred=cond_training_q,
                                                                        name='switch_actions_training_policy')
            (actions, _) = merge([sw_action_training_q, sw_action_training_policy])

          prev_layer = tf.concat([prev_layer, actions], axis=1)
        l += 1
        prev_layer = fully_connected(prev_layer, num_outputs=n_unit, activation_fn=activation,
                                     weights_initializer=initializer,
                                     weights_regularizer=regularizer,
                                     normalizer_fn=normalizer, #when specify norm , bias will be ignored.
                                     normalizer_params=norm_param,
                                     trainable=trainable)

      # end fc layers

      ##output layer. fully_connected will create bias which is not wanted in output layer.
      output_layer = fully_connected(inputs=prev_layer,num_outputs=1,
                                     activation_fn=None,
                                     weights_initializer=self.output_layer_initializer,
                                     weights_regularizer=self.output_layer_regularizer,
                                     biases_initializer=None, # to skip bias in output layer
                                    trainable=trainable)

    # == == end with variable_scope() ==
    return output_layer

  @property
  def online_q_outputs_tensor(self):
    return self.online_q_outputs

  @property
  def target_q_outputs_tensor(self):
    return self.target_q_outputs

  def compute_online_q_net_gradients(self, q_loss):
    grads_and_vars = self.optimizer.compute_gradients(
      q_loss, var_list=self.online_q_net_vars)
    grads = [g for (g, _) in grads_and_vars if g is not None]
    compute_op = tf.group(*grads)

    return (grads_and_vars, compute_op)


  def apply_online_q_net_gradients(self, grads_and_vars):
    vars_with_grad = [v for g, v in grads_and_vars if g is not None]
    if not vars_with_grad:
      raise ValueError(
        "$$ ddpg $$ q-net $$ No gradients provided for any variable, check your graph for ops"
        " that do not support gradients,variables %s." %
        ([str(v) for _, v in grads_and_vars]))
    return self.optimizer.apply_gradients(grads_and_vars)

  def soft_update_online_to_target(self):
    return soft_update_online_to_target(self.online_q_net_vars_by_name,
                                               self.target_q_net_vars_by_name)

  def copy_online_to_target(self):
    return copy_online_to_target(self.online_q_net_vars_by_name,
                                        self.target_q_net_vars_by_name)

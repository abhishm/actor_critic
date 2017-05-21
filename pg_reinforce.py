# import random
import numpy as np
import tensorflow as tf

class PolicyGradientREINFORCE(object):

  def __init__(self, session,
                     optimizer,
                     policy_network,
                     state_dim,
                     reg_param=0.001,      # regularization constants
                     max_gradient=5,       # max gradient norms
                     entropy_bonus=1,
                     summary_writer=None,
                     summary_every=100):

    # tensorflow machinery
    self.session        = session
    self.optimizer      = optimizer
    self.summary_writer = summary_writer
    self.summary_every  = summary_every
    self.no_op           = tf.no_op()

    # model components
    self.policy_network = policy_network
    self.state_dim = state_dim

    # training parameters
    self.max_gradient    = max_gradient
    self.reg_param       = reg_param
    self.entropy_bonus   = entropy_bonus

    #counter
    self.global_step = tf.Variable(0, name="global_step", trainable=False)

    # create and initialize variables
    self.create_variables()
    var_lists = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    self.session.run(tf.variables_initializer(var_lists))

    # make sure all variables are initialized
    self.session.run(tf.assert_variables_initialized())

    if self.summary_writer is not None:
      # graph was not available when journalist was created
      self.summary_writer.add_graph(self.session.graph)
      self.summary_every = summary_every

  def create_input_placeholders(self):
    with tf.name_scope("inputs"):
      self.states = tf.placeholder(tf.float32, (None, self.state_dim), name="states")
      self.actions = tf.placeholder(tf.int32, (None,), name="actions")
      self.returns = tf.placeholder(tf.float32, (None,), name="returns")

  def create_variables_for_actions(self):
    with tf.name_scope("generating_actions"):
      with tf.variable_scope("policy_network"):
        self.logit = self.policy_network(self.states)
      self.unnormalized_log_probs = tf.identity(self.logit, "unnormalized_log_probs")
      self.probs = tf.nn.softmax(self.unnormalized_log_probs)
      self.log_probs = tf.nn.log_softmax(self.unnormalized_log_probs)
    with tf.name_scope("computing_entropy"):
      self.entropy = tf.reduce_mean(tf.reduce_sum(tf.multiply(self.probs, -1.0 * self.log_probs),
                                                 reduction_indices=1))

  def create_variables_for_optimization(self):
    with tf.name_scope("optimization"):
      self.cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.unnormalized_log_probs,
                                                                          labels=self.actions)
      self.loss = tf.reduce_mean(tf.multiply(self.cross_entropy, self.returns))
      self.loss -= self.entropy_bonus * self.entropy
      self.trainable_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="policy_network")
      self.gradients = self.optimizer.compute_gradients(self.loss, var_list=self.trainable_variables)
      self.train_op = self.optimizer.apply_gradients(self.gradients, global_step=self.global_step)
      self.grad_norm = tf.global_norm([grad for grad, var in self.gradients])
      self.var_norm = tf.global_norm(self.trainable_variables)

  def create_summaries(self):
    self.loss_summary = tf.summary.scalar("policy/loss", self.loss)
    self.entropy_summary = tf.summary.scalar("policy/entropy", self.entropy)
    self.grad_norm_summary = tf.summary.scalar("policy/grad_norm", self.grad_norm)
    self.var_norm_summary = tf.summary.scalar("policy/var_norm", self.var_norm)

  def merge_summaries(self):
    self.summarize = tf.summary.merge([self.loss_summary,
                                       self.entropy_summary,
                                       self.grad_norm_summary,
                                       self.var_norm_summary])

  def create_variables(self):
    self.create_input_placeholders()
    self.create_variables_for_actions()
    self.create_variables_for_optimization()
    self.create_summaries()
    self.merge_summaries()

  def sampleAction(self, states):
    probs = self.session.run(self.probs, {self.states: states})[0]
    return np.argmax(np.random.multinomial(1, probs))

  def compute_action_probabilities(self, states):
    return self.session.run(self.probs, {self.states: states})

  def update_parameters(self, states, actions, returns):
    write_summary = self.train_itr % self.summary_every == 0
    _, summary = self.session.run([self.train_op,
                                   self.summarize if write_summary else self.no_op],
                                  {self.states: states,
                                   self.actions: actions,
                                   self.returns: returns})

    if write_summary:
        self.summary_writer.add_summary(summary, self.train_itr)

  @property
  def train_itr(self):
    return self.session.run(self.global_step)

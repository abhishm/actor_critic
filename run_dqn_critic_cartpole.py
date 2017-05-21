import tensorflow as tf
import numpy as np
import json
import gym
from tqdm import trange
import matplotlib.pyplot as plt
from dqn_agent import DQNAgent
from pg_reinforce import PolicyGradientREINFORCE
from replay_buffer import ReplayBuffer
from sampler import Sampler

config = json.load(open("configuration.json"))

# Environment parameters
env_name = config["env_name"]
env = gym.make(env_name)
state_dim   = env.observation_space.shape[0]
num_actions = env.action_space.n

# Policy nework parameters
entropy_bonus = config["entropy_bonus"]
policy_session = tf.Session()
policy_optimizer = tf.train.AdamOptimizer(learning_rate=config["policy_learning_rate"])
policy_writer = tf.summary.FileWriter("policy/")
policy_summary_every = 10

def show_image(array):
    """Create a pyplot plot and save to buffer."""
    plt.figure()
    plt.plot(array)
    plt.title("Reward Progress")
    plt.xlabel("Iteration number")
    plt.ylabel("rewards")
    plt.grid()
    plt.show()

def policy_network(states):
   """ define policy neural network """
   W1 = tf.get_variable("W1", [state_dim, 20],
                        initializer=tf.contrib.layers.xavier_initializer())
   b1 = tf.get_variable("b1", [20],
                        initializer=tf.constant_initializer(0))
   h1 = tf.nn.tanh(tf.matmul(states, W1) + b1)
   W2 = tf.get_variable("W2", [20, num_actions],
                        initializer=tf.contrib.layers.xavier_initializer())
   b2 = tf.get_variable("b2", [num_actions],
                        initializer=tf.constant_initializer(0))
   p = tf.matmul(h1, W2) + b2
   return p

pg_reinforce = PolicyGradientREINFORCE(policy_session,
                                       policy_optimizer,
                                       policy_network,
                                       state_dim,
                                       entropy_bonus=entropy_bonus,
                                       summary_writer=policy_writer,
                                       summary_every=policy_summary_every)

# Initializing Sampler
batch_size = config["batch_size"]
sampler = Sampler(pg_reinforce,
                  env,
                  config["batch_size"],
                  config["max_step"],
                  summary_writer=policy_writer)

# Q-network parameters
q_session = tf.Session()
q_optimizer = tf.train.AdamOptimizer(learning_rate=config["q_learning_rate"])
q_writer = tf.summary.FileWriter("q/")
q_summary_every = 10

# def action_masker(array):
#     masked_action = np.zeros((array.size, num_actions), dtype=np.float32)
#     masked_action[np.arange(array.size), array] = 1.0
#     return masked_action

def q_network(states):
    W1 = tf.get_variable("W1", [state_dim, 20],
                         initializer=tf.contrib.layers.xavier_initializer())
    b1 = tf.get_variable("b1", [20],
                         initializer=tf.constant_initializer(0))
    h1 = tf.nn.relu(tf.matmul(states, W1) + b1)
    W2 = tf.get_variable("W2", [20, num_actions],
                         initializer=tf.contrib.layers.xavier_initializer())
    b2 = tf.get_variable("b2", [num_actions],
                         initializer=tf.constant_initializer(0))
    q = tf.matmul(h1, W2) + b2
    return q

dqn_agent = DQNAgent(q_session,
                     q_optimizer,
                     q_network,
                     state_dim,
                     num_actions,
                     config["discount"],
                     config["target_update_rate"],
                     config["q_error_threshold"],
                     summary_writer=q_writer,
                     summary_every=q_summary_every)

# Initializing ReplayBuffer
buffer_size = config["buffer_size"]
sample_size = config["sample_size"]
q_network_updates = config["q_network_updates"]
replay_buffer = ReplayBuffer(buffer_size)

# Training
def computing_probabilities(batch):
    probabilites = pg_reinforce.compute_action_probabilities(batch["next_states"])
    return probabilites

def update_random_batch(batch):
    next_action_probs = computing_probabilities(batch)
    batch["next_action_probs"] = next_action_probs

def update_q_parameters():
    for _ in range(q_network_updates):
        random_batch = replay_buffer.sample_batch(sample_size)
        update_random_batch(random_batch)
        dqn_agent.update_parameters(random_batch)

def compute_return(batch):
    q_value = dqn_agent.compute_q_values(batch["states"], batch["actions"])
    # all_q_value = dqn_agent.compute_all_q_values(batch["states"])
    # probs = pg_reinforce.compute_action_probabilities(batch["states"])
    # v_value = np.sum(all_q_value * probs, axis=1)
    # adv = q_value - v_value
    return q_value #adv

reward = []
for _ in trange(config["num_itr"]):
    batch = sampler.collect_one_batch()
    replay_buffer.add_batch(batch)
    if sample_size <= replay_buffer.num_items:
        update_q_parameters()
        returns = compute_return(batch)
        pg_reinforce.update_parameters(batch["states"], batch["actions"], returns)
        reward.append(batch["rewards"].sum() / batch_size)

show_image(reward)

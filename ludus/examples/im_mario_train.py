import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPool2D, Flatten
from tensorflow.keras.backend import categorical_crossentropy
from ludus.policies import BaseTrainer
from ludus.env import EnvController
from ludus.utils import preprocess_atari, reshape_train_var
import gym
# Super Mario stuff
from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

name = 'im_mario-v1'

def make_env():
    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = BinarySpaceToDiscreteSpaceEnv(env, SIMPLE_MOVEMENT)
    return env

class IMTrainer(BaseTrainer):
    def __init__(self, in_op, out_op, value_out_op, act_type='discrete', sess=None, clip_val=0.2, ppo_iters=80,
                 target_kl=0.01, v_coef=1., entropy_coef=0.01):
        self.value_out_op = value_out_op
        self.clip_val = clip_val
        self.ppo_iters = ppo_iters
        self.target_kl = target_kl
        self.v_coef = v_coef
        self.entropy_coef = entropy_coef
        
        super().__init__(in_op, out_op, act_type, sess)
        
    def _create_ICM(self):
        feature_dim = 128
        r_i_scale = 2
        beta = 0.2
        
        # Create placeholder
        self.next_obs_holders = tf.placeholder(tf.float32, shape=self.in_op.shape)
        
        # Observation feature encoder
        with tf.variable_scope('feature_encoder'):
            self.f_obs = Conv2D(32, 4, (2, 2), activation=tf.nn.relu, name='fe_conv')(self.in_op)
            self.f_obs = MaxPool2D(2, name='fe_max_pool')(self.f_obs)
            self.f_obs = Conv2D(32, 3, activation=tf.nn.relu, name='fe_conv2')(self.f_obs)
            self.f_obs = MaxPool2D(2, name='fe_max_pool2')(self.f_obs)
            self.f_obs = Conv2D(32, 3, activation=tf.nn.relu, name='fe_conv3')(self.f_obs)
            self.f_obs = MaxPool2D(2, name='fe_max_pool3')(self.f_obs)
            self.f_obs = Flatten(name='fe_flattened')(self.f_obs)
            self.f_obs = Dense(feature_dim, activation=tf.nn.relu, name='fe_dense')(self.f_obs)
            
        with tf.variable_scope('feature_encoder', reuse=True):
            self.f_obs_next = Conv2D(32, 4, (2, 2), activation=tf.nn.relu, name='fe_conv')(self.in_op)
            self.f_obs_next = MaxPool2D(2, name='fe_max_pool')(self.f_obs_next)
            self.f_obs_next = Conv2D(32, 3, activation=tf.nn.relu, name='fe_conv2')(self.f_obs_next)
            self.f_obs_next = MaxPool2D(2, name='fe_max_pool2')(self.f_obs_next)
            self.f_obs_next = Conv2D(32, 3, activation=tf.nn.relu, name='fe_conv3')(self.f_obs_next)
            self.f_obs_next = MaxPool2D(2, name='fe_max_pool3')(self.f_obs_next)
            self.f_obs_next = Flatten(name='fe_flattened')(self.f_obs_next)
            self.f_obs_next = Dense(feature_dim, activation=tf.nn.relu, name='fe_dense')(self.f_obs_next)
            
        # State predictor forward model
        self.state_act_pair = tf.concat([self.act_masks, self.f_obs], axis=1)
        self.sp_dense = Dense(64, activation=tf.nn.relu)(self.state_act_pair)
        self.f_obs_next_pred = Dense(feature_dim, activation=tf.nn.relu)(self.sp_dense)
        
        # Inverse dynamics model (predicting action)
        self.state_state_pair = tf.concat([self.f_obs, self.f_obs_next], axis=1)
        self.act_preds = Dense(64, activation=tf.nn.relu)(self.state_state_pair)
        # TODO: softmax only works for discrete
        self.act_preds = Dense(self.out_op.shape[1], activation=tf.nn.softmax)(self.act_preds)
        
        # Calculating intrinsic reward
        self.obs_pred_diff = self.f_obs_next_pred - self.f_obs_next
        self.r_i = r_i_scale * tf.reduce_sum(self.obs_pred_diff ** 2, axis=1) # Fix these squares (Probably okay)
        
        # Calculating losses
        self.pre_loss_i = categorical_crossentropy(self.act_masks, self.act_preds) # tf.reduce_sum((self.act_holders - self.act_pred) ** 2, axis=1)
        self.pre_loss_f = tf.reduce_sum(self.obs_pred_diff ** 2, axis=1)
        
        self.loss_i = (1 - beta) * self.pre_loss_i
        self.loss_f = beta * self.pre_loss_f
        
    def _create_discrete_trainer(self, optimizer=tf.train.AdamOptimizer()):
        """
        Creates a function for vanilla policy training with a discrete action space
        """
        # First passthrough
        
        self.act_holders = tf.placeholder(tf.int32, shape=[None])
        self.reward_holders = tf.placeholder(tf.float32, shape=[None])
        
        self.act_masks = tf.one_hot(self.act_holders, self.out_op.shape[1].value, dtype=tf.float32)
        self.resp_acts = tf.reduce_sum(self.act_masks *  self.out_op, axis=1)
        
        self.advantages = self.reward_holders - tf.squeeze(self.value_out_op)
        
        self._create_ICM()
        
        # Second passthrough
        
        self.advatange_holders = tf.placeholder(dtype=tf.float32, shape=self.advantages.shape)
        self.old_prob_holders = tf.placeholder(dtype=tf.float32, shape=self.resp_acts.shape)
 
        self.policy_ratio = self.resp_acts / self.old_prob_holders
        self.clipped_ratio = tf.clip_by_value(self.policy_ratio, 1 - self.clip_val, 1 + self.clip_val)

        self.min_loss = tf.minimum(self.policy_ratio * self.advatange_holders, self.clipped_ratio * self.advatange_holders)
        
        self.optimizer = tf.train.AdamOptimizer()

        # Actor update
        
        self.kl_divergence = tf.reduce_mean(tf.log(self.old_prob_holders) - tf.log(self.resp_acts))
        self.actor_loss = -tf.reduce_mean(self.min_loss)
        self.actor_update = self.optimizer.minimize(self.actor_loss)

        # Value update
        
        self.value_loss = tf.reduce_mean(tf.square(self.reward_holders - tf.squeeze(self.value_out_op)))
        self.value_update = self.optimizer.minimize(self.value_loss)
        
        # Intrinsic motivation update
        
        self.intrinsic_loss = self.loss_i + self.loss_f
        self.intrinsic_update = self.optimizer.minimize(self.intrinsic_loss)
        
        # Combined update
        
        self.entropy = -tf.reduce_mean(tf.reduce_sum(self.out_op * tf.log(1. / tf.clip_by_value(self.out_op, 1e-8, 1.0)), axis=1))
        self.combined_loss = 0.1 * (self.actor_loss + self.v_coef * self.value_loss + self.entropy_coef * self.entropy) + self.intrinsic_loss
        self.combined_update = self.optimizer.minimize(self.combined_loss)
        
        def update_func(train_data):
            i_rews = []
            
            self.old_probs, self.old_advantages = self.sess.run([self.resp_acts, self.advantages], 
                                    feed_dict={self.in_op: reshape_train_var(train_data[:, 0]),
                                               self.act_holders: train_data[:, 1],
                                               self.reward_holders: train_data[:, 2]})
    
            for i in range(self.ppo_iters):
                kl_div, i_rew, _ = self.sess.run([self.kl_divergence, self.r_i, self.intrinsic_loss], 
                                   feed_dict={self.in_op: reshape_train_var(train_data[:, 0]),
                                        self.act_holders: reshape_train_var(train_data[:, 1]),
                                        self.reward_holders: train_data[:, 2],
                                        self.old_prob_holders: self.old_probs,
                                        self.advatange_holders: self.old_advantages})
                i_rews.append(i_rew)
                if kl_div > 1.5 * self.target_kl:
                    break    
    
            return np.mean(i_rews)

        self.sess.run(tf.global_variables_initializer())
        
        return update_func
        
    def _create_continuous_trainer(self):
        return

env = make_env() # This instance of the environment is only used
                              # to get action dimensions
in_shape = [84, 84, 1] # Size of reshaped observations

# Creating a conv net for the policy and value estimator
obs_op = Input(shape=in_shape)
conv1 = Conv2D(32, 8, (4, 4), activation='relu')(obs_op)
max_pool1 = MaxPool2D(2, 2)(conv1)
conv2 = Conv2D(32, 4, (2, 2), activation='relu')(max_pool1)
max_pool2 = MaxPool2D(2, 2)(conv2)
dense1 = Dense(256, activation='relu')(max_pool2)
flattened = Flatten()(dense1)

# Output probability distribution over possible actions
act_probs_op = Dense(env.action_space.n, activation='softmax')(flattened)

# Output value of observed state
value_op = Dense(1)(flattened)

# Wrap a Proximal Policy Optimization Trainer on top of the network
network = IMTrainer(obs_op, act_probs_op, value_op, act_type='discrete', ppo_iters=80)

saver = tf.train.Saver()

n_episodes = 20000 # Total episodes of data to collect
max_steps = 1024 # Max number of frames per game
batch_size = 6 # Smaller = faster, larger = stabler
print_freq = 5 # How many training updates between printing progress

# Create the environment controller for generating game data
ec = EnvController(make_env, n_threads=6)
# Set the preprocessing function for observations
ec.set_obs_transform(lambda x: preprocess_atari(x.squeeze()))

log_file = open(name + '.log', 'w+')

update_rewards = []
update_i_rewards = []

for i in range(int(n_episodes / batch_size)):
    ec.sim_episodes(network, batch_size, max_steps) # Simualate env to generate data
    update_rewards.append(ec.get_avg_reward()) # Append rewards to reward tracker list
    dat = ec.get_data() # Get all the data gathered
    i_rew = network.train(dat) # Train the network with PPO
    update_i_rewards.append(i_rew)
    if i != 0 and i % print_freq == 0:
        print(f'Update #{i}, Avg Reward (E, I): {np.mean(update_rewards[-print_freq:])}, {np.mean(update_i_rewards[-print_freq:])}') # Print an update
        log_file.write(f'Update #{i}, Avg Reward (E, I): {np.mean(update_rewards[-print_freq:])}, {np.mean(update_i_rewards[-print_freq:])}\n')
        log_file.flush()
        saver.save(network.sess, 'models/' + name + '.ckpt')

saver.save(network.sess, 'models/' + name + '.ckpt')
log_file.close()
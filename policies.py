import tensorflow as tf
import numpy as np
from ludus.utils import reshape_train_var, gaussian_likelihood

class BaseTrainer(object):
    def __init__(self, in_op, out_op, act_type='discrete', sess=None):
        if not sess:
            self.renew_sess()
        
        self.in_op = in_op
        self.out_op = out_op
        
        if act_type in ('discrete', 'd'):
            self.train = self._create_discrete_trainer()
            self.act_type = 'discrete'
        elif act_type in ('continuous', 'c'):
            self.train = self._create_continuous_trainer()
            self.act_type = 'continuous'
          
        if act_type not in ('discrete', 'd', 'continuous', 'c'):
            raise TypeError('act_type must be \'discrete\' or \'continuous\'')  
    
    def renew_sess(self):
        """
        Starts a new internal Tensorflow session
        """
        self.sess = tf.Session()
        
    def end_sess(self):
        """
        Ends the internal Tensorflow session if it exists
        """
        if self.sess:
            self.sess.close()
            
    def close(self):
        """
        Ends the internal Tensorflow session if it exists
        """
        if self.sess:
            self.sess.close()
            
    def _create_discrete_trainer(self):
        raise NotImplementedError('This class can only be used as a template')
        
    def _create_continuous_trainer(self):
        raise NotImplementedError('This class can only be used as a template')
            
    def _gen_discrete_act(self, obs):
        act_probs = self.sess.run(self.out_op, feed_dict={self.in_op: [obs]})
        act = np.random.choice(list(range(len(act_probs[0]))), p=act_probs[0])
        
        return act
    
    def _gen_continuous_act(self, obs):
        act_vect = self.sess.run(self.out_act, feed_dict={self.in_op: [obs]})[0]
        
        return np.array(act_vect)
        
    def gen_act(self, obs):
        if self.act_type == 'discrete':
            return self._gen_discrete_act(obs)
        else:
            return self._gen_continuous_act(obs)
        
    def train(self, obs, rewards, acts):
        raise RuntimeError('The train method was not properly created')

class VPGTrainer(BaseTrainer):
    def __init__(self, in_op, out_op, act_type='discrete', sess=None):
        """
        Create a wrapper for RL networks for easy training.
        Args:
            in_op (tf.Placeholder): Observation input to architecture
            out_op (tf.Variable): Action output of architecture
            act_type (string): 'discrete' for a discrete actions space or 'continuous'
                               for a continuous actions space
            sess (tf.Session): A session if you would like to use a custom session,
                               if left none it will be automatically created
        """
        super().__init__(in_op, out_op, act_type, sess)
        
    def _create_discrete_trainer(self, optimizer=tf.train.AdamOptimizer()):
        """
        Creates a function for vanilla policy training with a discrete action space
        """
        self.act_holders = tf.placeholder(tf.int32, shape=[None])
        self.reward_holders = tf.placeholder(tf.float32, shape=[None])
        
        self.act_masks = tf.one_hot(self.act_holders, self.out_op.shape[1].value, dtype=tf.float32)
        self.log_probs = tf.log(self.out_op)
        
        self.resp_acts = tf.reduce_sum(self.act_masks *  self.log_probs, axis=1)
        self.loss = -tf.reduce_mean(self.resp_acts * self.reward_holders)
        
        self.optimizer = optimizer
        self.update = self.optimizer.minimize(self.loss)
        
        update_func = lambda train_data: self.sess.run(self.update, 
                                                       feed_dict={self.in_op: reshape_train_var(train_data[:, 0]),
                                                            self.act_holders: reshape_train_var(train_data[:, 1]),
                                                            self.reward_holders: train_data[:, 2]})
        
        self.sess.run(tf.global_variables_initializer())
        
        return update_func
        
    def _create_continuous_trainer(self, optimizer=tf.train.AdamOptimizer()):
        """
        Creates a function for vanilla policy training with a continuous action space
        """
        self.act_holders = tf.placeholder(tf.float32, shape=[None, self.out_op.shape[1].value])
        self.reward_holders = tf.placeholder(tf.float32, shape=[None])
        
        self.std = tf.Variable(0.5 * np.ones(shape=self.out_op.shape[1].value), dtype=tf.float32)
        self.out_act = self.out_op + tf.random_normal(tf.shape(self.out_op), dtype=tf.float32) * self.std
        
        self.log_probs = gaussian_likelihood(self.act_holders, self.out_op, self.std)
        
        self.loss = -tf.reduce_mean(self.log_probs * self.reward_holders)
        
        self.optimizer = optimizer
        self.update = self.optimizer.minimize(self.loss)
        
        update_func = lambda train_data: self.sess.run(self.update, 
                                                       feed_dict={self.in_op: reshape_train_var(train_data[:, 0]),
                                                            self.act_holders: reshape_train_var(train_data[:, 1]),
                                                            self.reward_holders: train_data[:, 2]})
        
        self.sess.run(tf.global_variables_initializer())
        
        return update_func
        

class VAPGTrainer(BaseTrainer):
    def __init__(self, in_op, out_op, v_out_op, act_type='discrete', sess=None):
        self.v_out_op = v_out_op
        super().__init__(in_op, out_op, act_type, sess)
        
    def _create_discrete_trainer(self, optimizer=tf.train.AdamOptimizer()):
        """
        Creates a function for vanilla policy training with a discrete action space
        """
        self.act_holders = tf.placeholder(tf.int32, shape=[None])
        self.reward_holders = tf.placeholder(tf.float32, shape=[None])
        
        self.act_masks = tf.one_hot(self.act_holders, self.out_op.shape[1].value, dtype=tf.float32)
        self.log_probs = tf.log(self.out_op)
        
        self.advantages = self.reward_holders - self.v_out_op
        
        self.resp_acts = tf.reduce_sum(self.act_masks *  self.log_probs, axis=1)
        self.loss = -tf.reduce_mean(self.resp_acts * self.advantages)
        
        self.optimizer = optimizer
        self.actor_update = self.optimizer.minimize(self.loss)
        
        with tf.control_dependencies([self.actor_update]):
            self.value_loss = tf.reduce_mean(tf.square(self.reward_holders - tf.squeeze(self.v_out_op)))
            self.value_update = self.optimizer.minimize(self.value_loss)
        
        update_func = lambda train_data: self.sess.run([self.actor_update, self.value_update], 
                                                       feed_dict={self.in_op: reshape_train_var(train_data[:, 0]),
                                                            self.act_holders: reshape_train_var(train_data[:, 1]),
                                                            self.reward_holders: train_data[:, 2]})
        
        self.sess.run(tf.global_variables_initializer())
        
        return update_func
        
    def _create_continuous_trainer(self, optimizer=tf.train.AdamOptimizer()):
        """
        Creates a function for vanilla policy training with a continuous action space
        """
        self.act_holders = tf.placeholder(tf.float32, shape=[None, self.out_op.shape[1].value])
        self.reward_holders = tf.placeholder(tf.float32, shape=[None])
        
        self.std = tf.Variable(0.5 * np.ones(shape=self.out_op.shape[1].value), dtype=tf.float32)
        self.out_act = self.out_op + tf.random_normal(tf.shape(self.out_op), dtype=tf.float32) * self.std
        
        self.log_probs = gaussian_likelihood(self.act_holders, self.out_op, self.std)
        
        self.advantages = self.reward_holders - tf.squeeze(self.v_out_op)
        
        self.actor_loss = -tf.reduce_mean(self.log_probs * self.advantages)
        
        self.optimizer = optimizer
        self.actor_update = self.optimizer.minimize(self.actor_loss)
        
        with tf.control_dependencies([self.actor_update]):
            self.value_loss = tf.reduce_mean(tf.square(self.reward_holders - tf.squeeze(self.v_out_op)))
            self.value_update = self.optimizer.minimize(self.value_loss)
        
        update_func = lambda train_data: self.sess.run([self.actor_update, self.value_update], 
                                                       feed_dict={self.in_op: reshape_train_var(train_data[:, 0]),
                                                            self.act_holders: reshape_train_var(train_data[:, 1]),
                                                            self.reward_holders: train_data[:, 2]})
        
        self.sess.run(tf.global_variables_initializer())
        
        return update_func
    

class PPOTrainer(BaseTrainer):
    def __init__(self, in_op, out_op, value_out_op, act_type='discrete', sess=None, clip_val=0.2, ppo_iters=80,
                 target_kl=0.01, v_coef=1., entropy_coef=0.01):
        self.value_out_op = value_out_op
        self.clip_val = clip_val
        self.ppo_iters = ppo_iters
        self.target_kl = target_kl
        self.v_coef = v_coef
        self.entropy_coef = entropy_coef
        
        super().__init__(in_op, out_op, act_type, sess)
        
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
        
        # Combined update
        
        self.entropy = -tf.reduce_mean(tf.reduce_sum(self.out_op * tf.log(1. / tf.clip_by_value(self.out_op, 1e-8, 1.0)), axis=1))
        self.combined_loss = self.actor_loss + self.v_coef * self.value_loss + self.entropy_coef * self.entropy
        self.combined_update = self.optimizer.minimize(self.combined_loss)
        
        def update_func(train_data):
            self.old_probs, self.old_advantages = self.sess.run([self.resp_acts, self.advantages], 
                                    feed_dict={self.in_op: reshape_train_var(train_data[:, 0]),
                                               self.act_holders: train_data[:, 1],
                                               self.reward_holders: train_data[:, 2]})
        
            for i in range(self.ppo_iters):
                kl_div, _ = self.sess.run([self.kl_divergence, self.combined_update], 
                               feed_dict={self.in_op: reshape_train_var(train_data[:, 0]),
                                    self.act_holders: reshape_train_var(train_data[:, 1]),
                                    self.reward_holders: train_data[:, 2],
                                    self.old_prob_holders: self.old_probs,
                                    self.advatange_holders: self.old_advantages})
                if kl_div > 1.5 * self.target_kl:
                    break

        self.sess.run(tf.global_variables_initializer())
        
        return update_func
        
    def _create_continuous_trainer(self):
        """
        Creates a function for vanilla policy training with a continuous action space
        """
        # First passthrough
        
        self.act_holders = tf.placeholder(tf.float32, shape=[None, self.out_op.shape[1].value])
        self.reward_holders = tf.placeholder(tf.float32, shape=[None])
        
        self.std = tf.Variable(0.5 * np.ones(shape=self.out_op.shape[1].value), dtype=tf.float32)
        self.out_act = self.out_op + tf.random_normal(tf.shape(self.out_op), dtype=tf.float32) * self.std
        
        self.log_probs = gaussian_likelihood(self.act_holders, self.out_op, self.std)
        
        self.advantages = self.reward_holders - tf.squeeze(self.value_out_op)
        
        # Second passthrough
        
        self.advatange_holders = tf.placeholder(dtype=tf.float32, shape=self.advantages.shape)
        self.old_prob_holders = tf.placeholder(dtype=tf.float32, shape=self.log_probs.shape)
 
        self.policy_ratio = tf.exp(self.log_probs - self.old_prob_holders)
        self.clipped_ratio = tf.clip_by_value(self.policy_ratio, 1 - self.clip_val, 1 + self.clip_val)

        self.min_loss = tf.minimum(self.policy_ratio * self.advatange_holders, self.clipped_ratio * self.advatange_holders)
        
        self.optimizer = tf.train.AdamOptimizer()

        # Actor update
        
        self.kl_divergence = tf.reduce_mean(self.old_prob_holders - self.log_probs)
        self.actor_loss = -tf.reduce_mean(self.min_loss)
        self.actor_update = self.optimizer.minimize(self.actor_loss)

        # Value update
        
        self.value_loss = tf.reduce_mean(tf.square(self.reward_holders - tf.squeeze(self.value_out_op)))
        self.value_update = self.optimizer.minimize(self.value_loss)
        
        # Combined update
        
        self.entropy = -0.5 * tf.reduce_mean(tf.log(2 * np.pi * np.e * self.std))
        self.combined_loss = self.actor_loss + self.v_coef * self.value_loss + self.entropy_coef * self.entropy
        self.combined_update = self.optimizer.minimize(self.combined_loss)
        
        def update_func(train_data):
            self.old_probs, self.old_advantages = self.sess.run([self.log_probs, self.advantages], 
                                    feed_dict={self.in_op: reshape_train_var(train_data[:, 0]),
                                               self.act_holders: reshape_train_var(train_data[:, 1]),
                                               self.reward_holders: train_data[:, 2]})
            
            for i in range(self.ppo_iters):
                kl_div, _ = self.sess.run([self.kl_divergence, self.combined_update], 
                               feed_dict={self.in_op: reshape_train_var(train_data[:, 0]),
                                    self.act_holders: reshape_train_var(train_data[:, 1]),
                                    self.reward_holders: train_data[:, 2],
                                    self.old_prob_holders: self.old_probs,
                                    self.advatange_holders: self.old_advantages})
                if kl_div > 1.5 * self.target_kl:
                    break
            
            return kl_div, self.sess.run(self.entropy)

        self.sess.run(tf.global_variables_initializer())
        
        return update_func
import numpy as np
import tensorflow as tf


def running_avg(avg, new, update_rate=0.01):
    if avg is None:
        avg = new
    else:
        avg = (1 - update_rate) * avg + update_rate * new
    return avg


class Agent(object):

    def __init__(self, sess, pred_net, target_net, experience,
                 eps=0.1,
                 eps_min=0.0001,
                 eps_steps=10000,
                 learning_rate=0.001,
                 target_rate=1.0,
                 batch_size=32,
                 update_freq=4,
                 update_freq_target=1,
                 double_dqn=False,
                 discount=0.99,
                 nb_pretrain_step=None):

        self.sess = sess
        self.pred_net = pred_net
        self.target_net = target_net
        self.experience = experience
        self.eps = eps
        self.eps_min = eps_min
        self.eps_decay = (eps - eps_min) / eps_steps
        self.learning_rate = learning_rate
        self.target_rate = target_rate
        self.batch_size = batch_size
        self.double_dqn = double_dqn
        self.discount = discount
        self.update_freq = update_freq
        self.update_freq_target = update_freq_target
        if nb_pretrain_step is None:
            nb_pretrain_step = 0
        self.nb_pretrain_step = max(nb_pretrain_step, self.experience.size)
        self.define_loss_and_update()

    def define_loss_and_update(self):
        self.action = tf.placeholder(tf.int32, [None])
        self.target = tf.placeholder(tf.float32, [None])
        pred = tf.reduce_sum(
            self.pred_net.q_value * tf.one_hot(self.action,
                                               self.pred_net.nb_action),
            axis=1)
        self.loss = tf.reduce_mean(tf.squared_difference(pred, self.target))
        self.grads = tf.gradients(self.loss, self.pred_net.trainable_vars)
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.update_op = self.optimizer.apply_gradients(
            zip(self.grads, self.pred_net.trainable_vars))

        self.target_update = []
        for target_var, pred_var in zip(self.target_net.trainable_vars,
                                        self.pred_net.trainable_vars):
            assert target_var.name.split('/')[1:] == \
                pred_var.name.split('/')[1:]
            update = (1 - self.target_rate) * target_var.value() + \
                self.target_rate * pred_var.value()
            update = target_var.assign(update)
            self.target_update.append(update)

    def update(self):
        prestate, action, reward, poststate, terminal = \
            self.experience.sample(self.batch_size)

        target = self.sess.run(self.target_net.q_value,
                               feed_dict={self.target_net.state: poststate})
        if self.double_dqn:
            post_action = self.sess.run(
                self.pred_net.action,
                feed_dict={self.pred_net.state: poststate})
        else:
            post_action = np.argmax(target, axis=1)
        target = target[range(len(target)), post_action]
        target = reward + self.discount * target * (1 - terminal)

        loss, *_ = self.sess.run([self.loss, self.update_op],
                                 feed_dict={self.pred_net.state: prestate,
                                            self.action: action,
                                            self.target: target})

        self.eps = max(self.eps_min, self.eps - self.eps_decay)

        return loss

    def explore(self, env, nb_episode, max_steps=10000, callback=None):
        nb_step_tot = 0
        nb_update = 0
        nb_update_target = 0
        reward_avg = None
        loss_avg = None

        for episode in range(1, nb_episode + 1):
            state = env.reset()
            step = 0
            terminal = False
            reward_episode = 0

            while not terminal and step < max_steps:
                step += 1
                nb_step_tot += 1
                # Select action
                if nb_step_tot <= self.nb_pretrain_step or \
                        np.random.rand() < self.eps:
                    action = np.random.randint(0, self.pred_net.nb_action)
                else:
                    tmp = np.array([state])
                    action = self.sess.run(self.pred_net.action,
                                           feed_dict={self.pred_net.state: tmp})
                    action = action[0]

                # Take a step
                poststate, reward, terminal, info = env.step(action)
                self.experience.add(state, action, reward, poststate, terminal)
                reward_episode += reward

                # Update network
                if nb_step_tot > self.nb_pretrain_step:
                    if nb_step_tot % self.update_freq == 0:
                        nb_update += 1
                        loss = self.update()
                        loss_avg = running_avg(loss_avg, loss)
                    if nb_step_tot % self.update_freq_target == 0:
                        nb_update_target += 1
                        self.sess.run(self.target_update)

                if terminal:
                    reward_avg = running_avg(reward_avg, reward_episode)
                    if callback:
                        callback(episode=episode,
                                 nb_step_tot=nb_step_tot,
                                 nb_update=nb_update,
                                 nb_update_target=nb_update_target,
                                 reward_episode=reward_episode,
                                 reward_avg=reward_avg,
                                 loss_avg=loss_avg,
                                 eps=self.eps)

                state = poststate

    def play(self, env):
        pass

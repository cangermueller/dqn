import numpy as np
import tensorflow as tf


def running_avg(avg, new, update_rate=0.1):
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
                 discount=0.99,
                 nb_pretrain_step=None,
                 huber_loss=False,
                 max_grad_norm=None,
                 max_steps=10**4,
                 state_fun=None):
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
        self.discount = discount
        self.update_freq = update_freq
        self.update_freq_target = update_freq_target
        if nb_pretrain_step is None:
            nb_pretrain_step = 0
        self.nb_pretrain_step = max(nb_pretrain_step, self.experience.size)
        self.huber_loss = huber_loss
        self.max_grad_norm = max_grad_norm
        self.max_steps = max_steps
        self.state_fun = state_fun
        self.define_loss_and_update()

    def define_loss_and_update(self):
        self.action = tf.placeholder(tf.int32, [None])
        self.target = tf.placeholder(tf.float32, [None])
        pred = tf.reduce_sum(
            self.pred_net.q_value * tf.one_hot(self.action,
                                               self.pred_net.nb_action),
            axis=1)
        delta = tf.abs(pred - self.target)
        if self.huber_loss:
            self.loss = tf.where(delta < 1.0,
                                 0.5 * tf.square(delta),
                                 delta - 0.5)
        else:
            self.loss = tf.square(delta)
        self.loss = tf.reduce_mean(self.loss)
        self.grads = tf.gradients(self.loss, self.pred_net.trainable_vars)
        if self.max_grad_norm is not None:
            for idx, grad in enumerate(self.grads):
                self.grads[idx] = tf.clip_by_norm(grad, self.max_grad_norm)
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
        pass

    def get_action(self, state):
        state = np.array([state])
        action = self.sess.run(self.pred_net.action,
                               feed_dict={self.pred_net.state: state})
        action = action[0]
        return action

    def explore(self, env, nb_episode, callback=None):
        nb_step_tot = 0
        nb_update = 0
        nb_update_target = 0
        reward_avg = None
        target_avg = None
        q_value_avg = None
        loss_avg = None

        for episode in range(1, nb_episode + 1):
            observation = env.reset()
            if self.state_fun:
                state = self.state_fun(observation)
            else:
                state = observation
            step = 0
            terminal = False
            reward_episode = 0
            while not terminal and step < self.max_steps:
                step += 1
                nb_step_tot += 1
                # Select action
                if nb_step_tot <= self.nb_pretrain_step or \
                        np.random.rand() < self.eps:
                    action = np.random.randint(0, self.pred_net.nb_action)
                else:
                    action = self.get_action(state)

                # Take a step
                observation, reward, terminal, info = env.step(action)
                if self.state_fun:
                    poststate = self.state_fun(observation, state)
                else:
                    poststate = observation
                self.experience.add(state, action, reward, poststate, terminal)
                reward_episode += reward

                # Update network
                if nb_step_tot > self.nb_pretrain_step:
                    if nb_step_tot % self.update_freq == 0:
                        nb_update += 1
                        loss, target, q_value = self.update()
                        target_avg = running_avg(target_avg, target)
                        q_value_avg = running_avg(q_value_avg, q_value)
                        loss_avg = running_avg(loss_avg, loss)
                    if nb_step_tot % self.update_freq_target == 0:
                        nb_update_target += 1
                        self.sess.run(self.target_update)

                if terminal:
                    reward_avg = running_avg(reward_avg, reward_episode)
                    if callback:
                        callback(episode=episode,
                                 nb_step=step,
                                 nb_step_tot=nb_step_tot,
                                 nb_update=nb_update,
                                 nb_update_target=nb_update_target,
                                 reward_episode=reward_episode,
                                 reward_avg=reward_avg,
                                 target_avg=target_avg,
                                 q_value_avg=q_value_avg,
                                 loss_avg=loss_avg,
                                 eps=self.eps)

                state = poststate

    def play(self, env, nb_episode=1, log=print):
        for episode in range(1, nb_episode + 1):
            state = env.reset()
            env.render()
            step = 0
            terminal = False
            reward_episode = 0
            while not terminal and step < self.max_steps:
                step += 1
                action = self.get_action(state)
                state, reward, terminal, info = env.step(action)
                env.render()
                reward_episode += reward
            if log is not None:
                tmp = ['episode=%d' % episode,
                       'steps=%d' % step,
                       'reward=%.2f' % reward_episode]
                tmp = ' '.join(tmp)
                log(tmp)


class Dqn(Agent):

    def __init__(self, double_dqn=False, *args, **kwargs):
        self.double_dqn = double_dqn
        super(Dqn, self).__init__(*args, **kwargs)

    def update(self):
        prestate, action, reward, poststate, terminal = \
            self.experience.sample(self.batch_size)
        target = self.sess.run(self.target_net.q_value,
                               feed_dict={self.target_net.state: poststate})
        if self.double_dqn:
            postaction = self.sess.run(
                self.pred_net.action,
                feed_dict={self.pred_net.state: poststate})
        else:
            postaction = np.argmax(target, axis=1)
        target = target[range(len(target)), postaction]
        target = reward + self.discount * target * (1 - terminal)

        loss, q_value, *_ = self.sess.run(
            [self.loss, self.pred_net.q_value, self.update_op],
            feed_dict={self.pred_net.state: prestate,
                       self.action: action,
                       self.target: target})

        self.eps = max(self.eps_min, self.eps - self.eps_decay)

        return (loss, target.mean(), q_value.mean())

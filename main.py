#!/usr/bin/env python

from __future__ import division
from __future__ import print_function

import os
import sys

import argparse
import logging
import gym
import numpy as np
import tensorflow as tf

from dqn import agents, networks
from dqn.experience import Experience


def get_agent_class(agent):
    if agent == 'dqn':
        return agents.Dqn
    else:
        raise ValueError('Agent "%s" invalid!' % agent)


def pong_prepro_state(state, downsample=2):
    state = state[35:195]
    state = state[::downsample, ::downsample]
    state = state.astype(np.float32) / 256
    for idx in range(state.shape[-1]):
        state[:, :, idx] -= state[:, :, idx].mean()
    return state


def decorate_env(env, fun):
    env.reset = lambda _fun=env.reset: fun(_fun())

    def step(action, _step=env.step):
        tmp = list(_step(action))
        tmp[0] = fun(tmp[0])
        return tmp

    env.step = step


def count_params(variables):
    nb_param = 0
    for variable in variables:
        shape = variable.get_shape().as_list()
        nb_param += np.prod(shape)
    return nb_param


class App(object):

    def run(self, args):
        name = os.path.basename(args[0])
        parser = self.create_parser(name)
        opts = parser.parse_args(args[1:])
        return self.main(name, opts)

    def create_parser(self, name):
        p = argparse.ArgumentParser(
            prog=name,
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            description='Deep Q-learning')
        p.add_argument(
            '--env',
            help='Environment',
            default='FrozenLake-v0')
        p.add_argument(
            '-i', '--in_checkpoint',
            help='Input checkpoint path')
        p.add_argument(
            '-o', '--out_checkpoint',
            help='Output checkpoint path')
        p.add_argument(
            '--save_freq',
            help='Frequency in epochs to create checkpoint',
            type=int,
            default=1000)

        p.add_argument(
            '--nb_episode',
            help='Number of episodes',
            type=int,
            default=100)
        p.add_argument(
            '--nb_play',
            help='Number of episodes to play',
            type=int,
            default=0)

        # Learning parameters
        p.add_argument(
            '--agent',
            help='Name of agent',
            choices=['dqn'],
            default='dqn')
        p.add_argument(
            '--learning_rate',
            help='Learning rate',
            type=float,
            default=0.001)
        p.add_argument(
            '--target_rate',
            help='Learning rate of target network for DDQN',
            type=float,
            default=0.001)
        p.add_argument(
            '--batch_size',
            help='Batch size',
            type=int,
            default=16)
        p.add_argument(
            '--double_dqn',
            help='Use double DQN',
            action='store_true')
        p.add_argument(
            '--discount',
            help='Reward discount factors',
            type=float,
            default=0.99)
        p.add_argument(
            '--experience_size',
            help='Size of experience replay buffer',
            type=int,
            default=1000)
        p.add_argument(
            '--nb_pretrain_step',
            help='Number of pretraining steps',
            type=int,
            default=1000)
        p.add_argument(
            '--update_freq',
            help='Update frequency in steps of prediction network',
            type=int,
            default=4)
        p.add_argument(
            '--update_freq_target',
            help='Update frequency of target network as multiple for ' +
            '`update_freq`',
            type=int,
            default=1)
        p.add_argument(
            '--eps',
            help='Start value of eps parameter',
            type=float,
            default=0.1)
        p.add_argument(
            '--eps_min',
            help='Minimum of eps parameter',
            type=float,
            default=0.001)
        p.add_argument(
            '--eps_steps',
            help='Number of eps annealing steps',
            type=int,
            default=1000)
        p.add_argument(
            '--huber_loss',
            help='Use Huber loss',
            action='store_true')
        p.add_argument(
            '--max_grad_norm',
            help='Maximum gradient norm',
            type=float)

        # Network architecture
        p.add_argument(
            '--no_dual',
            help='No dual architecture',
            action='store_true')
        p.add_argument(
            '--nb_hidden',
            help='Number of hidden units in MLP',
            type=int,
            nargs='+',
            default=[10])
        p.add_argument(
            '--nb_kernel',
            help='Number of kernels in CNN',
            type=int,
            nargs='+',
            default=[32, 64])
        p.add_argument(
            '--kernel_sizes',
            help='Kernels sizes in CNN',
            type=int,
            nargs='+',
            default=[3, 3])
        p.add_argument(
            '--pool_sizes',
            help='Pooling sizes in CNN',
            type=int,
            nargs='+',
            default=[32, 64])
        p.add_argument(
            '--dropout',
            help='Dropout rate',
            type=float,
            default=0.1)

        # Misc
        p.add_argument(
            '--verbose',
            help='More detailed log messages',
            action='store_true')
        p.add_argument(
            '--log_file',
            help='Write log messages to file')

        return p

    def callback(self, episode, nb_step_tot,
                 nb_update, nb_update_target,
                 reward_episode, reward_avg, loss_avg,
                 eps, *args, **kwargs):

        if episode % self.opts.save_freq == 0:
            self.save_graph()

        def format_na(x, spec):
            if x is None:
                return 'NA'
            else:
                return '{:{spec}}'.format(x, spec=spec)

        tmp = ['episode={episode:d}',
               'steps={steps:d}',
               'updates={updates:d}',
               'updates_t={updates_t:d}',
               'r_epi={r_epi:.2f}',
               'r_avg={r_avg:s}',
               'loss={loss:s}',
               'eps={eps:.4f}']
        tmp = '  '.join(tmp)
        tmp = tmp.format(episode=episode,
                         steps=nb_step_tot,
                         updates=nb_update,
                         updates_t=nb_update_target,
                         r_epi=reward_episode,
                         r_avg=format_na(reward_avg, '.2f'),
                         loss=format_na(loss_avg, '5g'),
                         eps=eps)
        print(tmp)

    def save_graph(self):
        if self.opts.out_checkpoint:
            self.log.info('Saving graph to %s ...' % self.opts.out_checkpoint)
            self.saver.save(self.sess, self.opts.out_checkpoint)

    def main(self, name, opts):
        logging.basicConfig(filename=opts.log_file,
                            format='%(levelname)s (%(asctime)s): %(message)s')
        log = logging.getLogger(name)
        if opts.verbose:
            log.setLevel(logging.DEBUG)
        else:
            log.setLevel(logging.INFO)
        log.debug(opts)

        self.opts = opts
        self.log = log

        env = gym.make(opts.env)

        # Build networks
        if isinstance(env.observation_space, gym.spaces.Box) and \
                len(env.observation_space.shape) == 3:
            network_class = networks.Cnn
        else:
            network_class = networks.Mlp

        max_steps = 1000
        if opts.env == 'Pong-v0':
            state_shape = [80, 80, 3]
            state = tf.placeholder(tf.float32, [None] + state_shape,
                                   name='state')
            prepro_state = state
            decorate_env(env, pong_prepro_state)
            max_steps = 1500
        elif isinstance(env.observation_space, gym.spaces.Discrete):
            state_shape = None
            state = tf.placeholder(tf.int32, [None], name='state')
            prepro_state = tf.one_hot(state, env.observation_space.n)
        else:
            state_shape = list(env.observation_space.shape)
            state = tf.placeholder(tf.float32, [None] + state_shape,
                                   name='state')
            prepro_state = state

        nets = []
        for name in ['pred_net', 'target_net']:
            log.info('Building %s ...' % name)
            with tf.variable_scope(name):
                nets.append(network_class(state=state,
                                          nb_action=env.action_space.n,
                                          prepro_state=prepro_state,
                                          dual=not opts.no_dual,
                                          nb_hidden=opts.nb_hidden,
                                          nb_kernel=opts.nb_kernel,
                                          kernel_sizes=opts.kernel_sizes,
                                          pool_sizes=opts.pool_sizes,
                                          dropout=opts.dropout))
        pred_net, target_net = nets
        log.info('Number of network parameters: %d' %
                 count_params(pred_net.trainable_vars))

        # Setup agent
        experience = Experience(opts.experience_size, state_shape=state_shape)
        self.sess = tf.Session()
        agent_class = get_agent_class(opts.agent)
        agent = agent_class(sess=self.sess,
                            pred_net=pred_net,
                            target_net=target_net,
                            experience=experience,
                            eps=opts.eps,
                            eps_min=opts.eps_min,
                            eps_steps=opts.eps_steps,
                            learning_rate=opts.learning_rate,
                            target_rate=opts.target_rate,
                            batch_size=opts.batch_size,
                            double_dqn=opts.double_dqn,
                            discount=opts.discount,
                            update_freq=opts.update_freq,
                            update_freq_target=opts.update_freq *
                            opts.update_freq_target,
                            nb_pretrain_step=opts.nb_pretrain_step,
                            huber_loss=opts.huber_loss,
                            max_steps=max_steps,
                            max_grad_norm=opts.max_grad_norm,
                            )

        self.saver = tf.train.Saver()
        if opts.in_checkpoint:
            log.info('Restoring graph from %s ...' % opts.in_checkpoint)
            self.saver.restore(self.sess, opts.in_checkpoint)
        else:
            self.sess.run(tf.global_variables_initializer())

        if opts.nb_episode:
            agent.explore(env, opts.nb_episode, callback=self.callback)

        if opts.nb_play:
            agent.play(env, opts.nb_play)

        self.save_graph()
        return 0


if __name__ == '__main__':
    app = App()
    app.run(sys.argv)

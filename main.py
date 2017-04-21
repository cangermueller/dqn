#!/usr/bin/env python

from __future__ import division
from __future__ import print_function

import os
import sys

import argparse
import logging
import gym
import tensorflow as tf

from dqn import agents, networks
from dqn.experience import Experience


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
            '-i', '--in_model',
            help='Input model checkpoint path')
        p.add_argument(
            '-o', '--out_model',
            help='Output model checkpoint path')
        p.add_argument(
            '--save_freq',
            help='Frequency in epochs to save models',
            type=int,
            default=1000)

        p.add_argument(
            '--nb_episode',
            help='Number of episodes',
            type=int,
            default=100)

        # Learning parameters
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
            help='Update frequency in steps of target network',
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
            default=0.00001)
        p.add_argument(
            '--eps_steps',
            help='Number of eps annealing steps',
            type=int,
            default=1000)

        # Network architecture
        p.add_argument(
            '--nb_hidden',
            help='Number of hidden units in MLP',
            type=int,
            nargs='+',
            default=[10])

        # Misc
        p.add_argument(
            '--verbose',
            help='More detailed log messages',
            action='store_true')
        p.add_argument(
            '--log_file',
            help='Write log messages to file')

        return p

    def build_cnn(self, *args, **kwargs):
        return networks.Cnn(*args, **kwargs)

    def build_mlp(self, *args, **kwargs):
        opts = self.opts
        return networks.Mlp(nb_hidden=opts.nb_hidden, *args, **kwargs)

    def log_explore(self, episode, nb_step_tot, reward_episode, reward_avg,
                    loss_avg):

        def format_na_float(x):
            if x is None:
                return 'NA'
            else:
                return '%.2f' % x

        tmp = ['episode={episode:d}',
               'steps={steps:d}',
               'r_epi={r_epi:.2f}',
               'r_avg={r_avg:s}',
               'loss={loss:s}']
        tmp = '  '.join(tmp)
        tmp = tmp.format(episode=episode,
                         steps=nb_step_tot,
                         r_epi=reward_episode,
                         r_avg=format_na_float(reward_avg),
                         loss=format_na_float(loss_avg))
        print(tmp)

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
                len(env.observation_space.shape) == 2:
            network_fun = self.build_cnn
        else:
            network_fun = self.build_mlp
        if isinstance(env.observation_space, gym.spaces.Discrete):
            state_shape = None
            state = tf.placeholder(tf.int32, [None], name='state')
            prepro_state = tf.one_hot(state, env.observation_space.n)
        else:
            state_shape = list(env.observation_space.shape)
            state = tf.placeholder(tf.float32, [None] + state_shape,
                                   name='state')
            prepro_state = state

        pred_net = network_fun(state=state, nb_action=env.action_space.n,
                               prepro_state=prepro_state)
        target_net = network_fun(state=state, nb_action=env.action_space.n,
                                 prepro_state=prepro_state)

        # Setup agent
        experience = Experience(opts.experience_size, state_shape=state_shape)
        sess = tf.Session()
        agent = agents.Agent(sess, pred_net, target_net, experience,
                             eps=opts.eps,
                             eps_min=opts.eps_min,
                             eps_steps=opts.eps_steps,
                             learning_rate=opts.learning_rate,
                             target_rate=opts.target_rate,
                             batch_size=opts.batch_size,
                             double_dqn=opts.double_dqn,
                             discount=opts.discount,
                             update_freq=opts.update_freq,
                             update_freq_target=opts.update_freq_target,
                             nb_pretrain_step=opts.nb_pretrain_step)

        sess.run(tf.global_variables_initializer())
        agent.explore(env, opts.nb_episode, callback=self.log_explore)

        return 0


if __name__ == '__main__':
    app = App()
    app.run(sys.argv)

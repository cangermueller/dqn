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
            '-v', '--env',
            help='Environment',
            default='CartPole-v0')
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
            '-e', '--nb_episode',
            help='Number of episodes',
            type=int,
            default=100)
        p.add_argument(
            '--nb_pretrain_step',
            help='Number of pretraining steps',
            type=int,
            default=1000)
        p.add_argument(
            '--buffer_size',
            help='Size of replay buffer',
            type=int,
            default=1000)
        p.add_argument(
            '--eps_start',
            help='Start value of epsilon parameter',
            type=float,
            default=0.1)
        p.add_argument(
            '--eps_end',
            help='End value of epsilon parameter',
            type=float,
            default=0.00001)
        p.add_argument(
            '--nb_eps_step',
            help='Number of epsilon annealing steps',
            type=int,
            default=1000)
        p.add_argument(
            '-l', '--learning_rate',
            help='Learning rate',
            type=float,
            default=0.001)
        p.add_argument(
            '--ddqn',
            help='Use double DQN',
            action='store_true')
        p.add_argument(
            '--target_rate',
            help='Learning rate of target network for DDQN',
            type=float,
            default=0.001)
        p.add_argument(
            '-b', '--batch_size',
            help='Batch size',
            type=int,
            default=16)
        p.add_argument(
            '--nb_hidden',
            help='Number of hidden units',
            type=int,
            default=10)
        p.add_argument(
            '--log_freq',
            help='Logging frequency',
            type=int,
            default=1)
        p.add_argument(
            '--render',
            help='Number of episodes to render at the end',
            type=int,
            default=0)
        p.add_argument(
            '--verbose',
            help='More detailed log messages',
            action='store_true')
        p.add_argument(
            '--log_file',
            help='Write log messages to file')
        return p

    def main(self, name, opts):
        logging.basicConfig(filename=opts.log_file,
                            format='%(levelname)s (%(asctime)s): %(message)s')
        log = logging.getLogger(name)
        if opts.verbose:
            log.setLevel(logging.DEBUG)
        else:
            log.setLevel(logging.INFO)
        log.debug(opts)


        return 0


if __name__ == '__main__':
    app = App()
    app.run(sys.argv)

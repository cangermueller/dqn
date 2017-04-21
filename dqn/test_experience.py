import numpy as np
import pytest

from .experience import Experience


def add_random(exp):
    state_shape = exp.state_shape
    prestate = np.random.uniform(0, 1, state_shape)
    action = np.random.randint(0, 10)
    reward = np.random.rand()
    poststate = np.random.uniform(0, 1, state_shape)
    terminal = np.random.randint(0, 1)
    exp.add(prestate, action, reward, poststate, terminal)


def check_shape(count, state_shape, prestates, actions, rewards, poststates,
               terminals):
    assert prestates.shape == (count, state_shape)
    assert poststates.shape == (count, state_shape)
    assert actions.shape == (count,)
    assert rewards.shape == (count,)
    assert terminals.shape == (count,)


class TestExperience(object):

    def test(self):
        state_shape = 4
        size = 10
        exp = Experience(size, [state_shape])

        for i in range(size - 7):
            add_random(exp)
        with pytest.raises(ValueError):
            exp.sample(1)

        for i in range(7):
            add_random(exp)
        with pytest.raises(ValueError):
            exp.sample(size + 1)

        for i in range(1, size):
            prestates, actions, rewards, poststates, terminals = \
                exp.sample(i)
            #  print(prestates)
            #  print(actions)
            check_shape(i, state_shape,
                        prestates, actions, rewards, poststates, terminals)

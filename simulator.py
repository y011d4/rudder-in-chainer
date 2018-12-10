import numpy as np
from absl import app, flags, logging

FLAGS = flags.FLAGS


class Simulator():

    def __init__(self):
        if FLAGS.gpu>=0:
            import cupy
            self.xp = cupy
        else:
            self.xp = np

    def generate_sample(self):
        xp = self.xp
        """Create sample episodes from our example environment"""
        # Create random actions
        actions = np.asarray(np.random.randint(
            low=0, high=2, size=(FLAGS.max_timestep,)), dtype=np.float32)
        actions_onehot = np.zeros((FLAGS.max_timestep, 2), dtype=np.float32)
        actions_onehot[actions == 0, 0] = 1
        actions_onehot[:, 1] = 1 - actions_onehot[:, 0]  # [0, 1] or [1, 0]
        actions += actions - 1  # -1 or 1

        # Create states to actions, make sure agent stays in range [-6, 6]
        states = np.zeros_like(actions)
        for i, a in enumerate(actions):
            if i == 0:
                states[i] = a
            else:
                states[i] = np.clip(states[i-1] + a, a_min=-
                                    int(FLAGS.n_features/2), a_max=int(FLAGS.n_features/2))

        # Check when agent collected a coin (=is at position 2)
        coin_collect = np.asarray(states == 2, dtype=np.float32)
        #coin_collect[states == -3] = -1

        # Move all reward to position 50 to make it a delayed reward example
        true_rewards = coin_collect * 1.0
        true_rewards = np.concatenate((np.zeros_like(
            true_rewards[:FLAGS.n_padding_frame]), true_rewards, np.zeros_like(true_rewards[:FLAGS.n_padding_frame])))
        coin_collect[-1] = np.sum(coin_collect)
        coin_collect[:-1] = 0
        rewards = coin_collect

        # Padd end of game sequences with zero-states
        states = np.asarray(states, np.int) + int(FLAGS.n_features/2)
        states_onehot = np.zeros(
            (len(rewards)+FLAGS.n_padding_frame, FLAGS.n_features), dtype=np.float32)
        states_onehot[np.arange(len(rewards)), states] = 1
        states_onehot = np.concatenate(
            (np.zeros_like(states_onehot[:FLAGS.n_padding_frame, :]), states_onehot), axis=0)
        actions_onehot = np.concatenate(
            (np.zeros_like(actions_onehot[:FLAGS.n_padding_frame]), actions_onehot, np.zeros_like(actions_onehot[:FLAGS.n_padding_frame])))
        rewards = np.concatenate((np.zeros_like(
            rewards[:FLAGS.n_padding_frame]), rewards, np.zeros_like(rewards[:FLAGS.n_padding_frame])))
        # Return states, actions, and rewards
        return dict(states=xp.array(states_onehot[None, :]), actions=xp.array(actions_onehot[None, :]), rewards=xp.array(rewards[None, :, None]), true_rewards=xp.array(true_rewards[None, :, None]))

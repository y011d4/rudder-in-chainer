import os
import random
import datetime as dt
from collections import OrderedDict, deque
import numpy as np
import matplotlib.pyplot as plt
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import Variable
from absl import app, flags, logging

FLAGS = flags.FLAGS

flags.DEFINE_integer('load_epoch', 0, 'load model and optimizer')
flags.DEFINE_integer('load_trainer', 0, 'load trainer', short_name='l')
flags.DEFINE_integer('random_seed', 0, 'Random seed')
flags.DEFINE_integer('n_lstm', 32, 'n_lstm')
flags.DEFINE_integer('n_out', 1, 'n_out')

max_timestep = 50
n_mb = 1
n_features = 13
n_actions = 2
ending_frames = 10

def reset_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    if chainer.cuda.available:
        chainer.cuda.cupy.random.seed(seed)

def generate_sample():
    """Create sample episodes from our example environment"""
    # Create random actions
    actions = np.asarray(np.random.randint(
        low=0, high=2, size=(max_timestep,)), dtype=np.float32)
    actions_onehot = np.zeros((max_timestep, 2), dtype=np.float32)
    actions_onehot[actions == 0, 0] = 1
    actions_onehot[:, 1] = 1 - actions_onehot[:, 0] # [0, 1] or [1, 0]
    actions += actions - 1 # -1 or 1

    # Create states to actions, make sure agent stays in range [-6, 6]
    states = np.zeros_like(actions)
    for i, a in enumerate(actions):
        if i == 0:
            states[i] = a
        else:
            states[i] = np.clip(states[i-1] + a, a_min=-
                                int(n_features/2), a_max=int(n_features/2))

    # Check when agent collected a coin (=is at position 2)
    coin_collect = np.asarray(states == 3, dtype=np.float32)
    coin_collect[states==-3] = -1

    # Move all reward to position 50 to make it a delayed reward example
    true_rewards = coin_collect * 1.0
    true_rewards = np.concatenate((np.zeros_like(true_rewards[:ending_frames]), true_rewards, np.zeros_like(true_rewards[:ending_frames])))
    coin_collect[-1] = np.sum(coin_collect)
    coin_collect[:-1] = 0
    rewards = coin_collect

    # Padd end of game sequences with zero-states
    states = np.asarray(states, np.int) + int(n_features/2)
    states_onehot = np.zeros(
        (len(rewards)+ending_frames, n_features), dtype=np.float32)
    states_onehot[np.arange(len(rewards)), states] = 1
    states_onehot = np.concatenate((np.zeros_like(states_onehot[:10, :]), states_onehot), axis=0)
    actions_onehot = np.concatenate(
            (np.zeros_like(actions_onehot[:ending_frames]), actions_onehot, np.zeros_like(actions_onehot[:ending_frames])))
    rewards = np.concatenate((np.zeros_like(rewards[:ending_frames]), rewards, np.zeros_like(rewards[:ending_frames])))
    # Return states, actions, and rewards
    return dict(states=states_onehot[None, :], actions=actions_onehot[None, :], rewards=rewards[None, :, None], true_rewards=true_rewards[None, :, None])

def create_computational_graph(variable, filename='./graph.dot'):
    g = chainer.computational_graph.build_computational_graph(variable)
    with open(filename, 'w') as o:
        o.write(g.dump())

class RewardRedistributionModel(chainer.Chain):

    def __init__(self):
        super(RewardRedistributionModel, self).__init__()
        with self.init_scope():
            self.lstm = L.LSTM(None, FLAGS.n_lstm, lateral_init=chainer.initializers.Normal(scale=1.0))
            self.l = L.Linear(None, FLAGS.n_out)

    def __call__(self, x):
        h = self.lstm(x)
        return self.l(h)

    def reset_state(self):
        self.lstm.reset_state()

    def return_decomposition(self, epoch):
        sample = generate_sample()
        lstm_inputs = F.concat((sample['states'], sample['actions']), axis=2)
        n_intgrd_steps = 500
        input = F.concat([lstm_inputs * w for w in np.linspace(0.0, 1.0, n_intgrd_steps)], axis=0)

        # Re-define input for gradient calculation
        input = Variable(input.data)

        for i in range(70):
            pred = self(input[:, i:i+1, :])
        intgrd_pred = pred[:, 0:1]

        if epoch==1:
            create_computational_graph(intgrd_pred, filename='./graph_intg.dot')

        self.reset_state()
        self.cleargrads()
        intgrd_pred.grad = np.ones((500, 1), dtype='f')
        intgrd_pred.backward(retain_grad=True)
        intgrd_pred.unchain_backward()

        grads = input.grad
        grads = np.where(np.isnan(grads), np.zeros_like(grads), grads)
        intgrd_grads = np.sum(grads, axis=0)
        intgrd_grads *= lstm_inputs[0].data
        intgrd_grads = np.sum(intgrd_grads, axis=-1) / n_intgrd_steps
        intgrd_grads = np.concatenate([np.zeros_like(intgrd_grads[:10]), intgrd_grads[10:-10], np.zeros_like(intgrd_grads[:10])], axis=0)

        intgrd_zero_prediction = intgrd_pred[0]
        intgrd_full_prediction = intgrd_pred[-1]
        intgrd_prediction_diff = intgrd_full_prediction - intgrd_zero_prediction
        intgrd_sum = np.sum(intgrd_grads)
        #intgrd_grads *= np.sum(sample['true_rewards'][0, :, 0])/intgrd_sum

        plt.plot(intgrd_grads.data[10:60], label='pred')
        plt.plot(sample['true_rewards'][0, 10:60, 0], label='true')
        plt.legend(loc='best')
        plt.xlim(0.0, 50.0)
        plt.ylim(-1.2, 1.2)
        plt.xlabel('Time step')
        plt.ylabel('Reward')
        #plt.show()
        plt.savefig('./result/rudder_{}.png'.format(epoch))
        plt.clf()


class Iterator(chainer.dataset.iterator.Iterator):
    def __init__(self):
        self.epoch = 0

    def __next__(self):
        self.epoch += 1
        return self.epoch

    @property
    def epoch_detail(self):
        return self.epoch


class Updater(chainer.training.updaters.StandardUpdater):

    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.model = optimizer.target
        super(Updater, self).__init__(Iterator(), optimizer)
        self.loss_history = deque(maxlen=100)
        self.min_loss = 10000000.0

    def update_LSTM(self, epoch):
        #self.optimizer.target.reset_state()
        sample = generate_sample()
        input = F.concat((sample['states'], sample['actions']), axis=2)
        input = Variable(input.data)

        pred = []
        for i in range(70):
            pred.append(self.model(input[:, i:i+1]))
        pred = F.stack(pred, axis=1)
        true_return = F.sum(Variable(sample['rewards']))
        loss = F.square(true_return-pred[:, -1, 0])

        self.optimizer.target.reset_state()
        self.optimizer.target.cleargrads()
        loss.backward()
        loss.unchain_backward()
        self.optimizer.update()

        loss = loss.data[0]

        chainer.report(
                {'loss': loss, 'pred': pred[0, -1, 0].data, 'actual': true_return.data}, observer=self.optimizer.target)

        return sample, loss


    def update_core(self):
        #opt = self.get_optimizer('main')
        epoch = self.get_iterator('main').next()

        sample, loss = self.update_LSTM(epoch)

        self.loss_history.append(loss)
        mean_loss = np.mean(self.loss_history)

        if mean_loss<self.min_loss and len(self.loss_history)==100:
            self.min_loss = mean_loss
            logging.info('min_loss is updated: %.5f' % self.min_loss)
            self.model.return_decomposition(epoch)


def main(argv):
    reset_seed(seed=FLAGS.random_seed)

    model = RewardRedistributionModel()
    optimizer = chainer.optimizers.Adam(alpha=0.01)
    optimizer.setup(model)

    #if FLAGS.load_epoch:
    #    chainer.serializers.load_npz('./result/model_snapshot_{}'.format(FLAGS.load_epoch), model)
    #    chainer.serializers.load_npz('./result/optimizer_snapshot_{}'.format(FLAGS.load_epoch), optimizer)

    updater = Updater(optimizer)

    trainer = chainer.training.Trainer(updater, (100000, 'epoch'))

    if FLAGS.load_trainer:
        chainer.serializers.load_npz('./result/snapshot_iter_{}'.format(FLAGS.load_trainer), trainer)
        updater.get_iterator('main').epoch = FLAGS.load_trainer

    trainer.extend(chainer.training.extensions.LogReport(trigger=(100, 'epoch')))
    trainer.extend(chainer.training.extensions.ParameterStatistics(model))
    trainer.extend(chainer.training.extensions.PrintReport(
        ['epoch', 'main/loss', 'main/pred', 'main/actual', 'elapsed_time']))
    #trainer.extend(chainer.training.extensions.ProgressBar())
    trainer.extend(chainer.training.extensions.snapshot_object(model, 'model_snapshot_{.updater.epoch}'), trigger=(1000, 'epoch'))
    trainer.extend(chainer.training.extensions.snapshot_object(optimizer, 'optimizer_snapshot_{.updater.epoch}'), trigger=(1000, 'epoch'))
    trainer.extend(chainer.training.extensions.snapshot(), trigger=(1000, 'epoch'))

    trainer.run()

if __name__ == '__main__':
    app.run(main)

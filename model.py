import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
from absl import app, flags, logging

FLAGS = flags.FLAGS


class LSTMAndFC(chainer.Chain):

    def __init__(self):
        super(LSTMAndFC, self).__init__()
        with self.init_scope():
            self.lstm = L.LSTM(
                None, FLAGS.n_lstm, lateral_init=chainer.initializers.Normal(scale=1.0))
            self.l = L.Linear(None, FLAGS.n_out)

    def __call__(self, x):
        h = self.lstm(x)
        return self.l(h)

    def reset_state(self):
        self.lstm.reset_state()

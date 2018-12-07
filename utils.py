import random
import numpy as np
import chainer
from absl import app, flags, logging

FLAGS = flags.FLAGS


def reset_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    if chainer.cuda.available:
        chainer.cuda.cupy.random.seed(seed)


def create_computational_graph(variable, filename='./graph.dot'):
    g = chainer.computational_graph.build_computational_graph(variable)
    with open(filename, 'w') as o:
        o.write(g.dump())


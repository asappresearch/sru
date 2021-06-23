import random

import numpy as np
import torch


def calc_norm(lis):
    l2_sum = sum(x.norm()**2 for x in lis)
    return l2_sum**0.5


def copy_model(model):
    states = model.state_dict()
    for k in states:
        v = states[k]
        states[k] = v.clone().cpu()
    return states


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def log_stats(writer, niter, model, memory):
    for i, x_i in enumerate(memory):
        if writer is not None:
            writer.add_scalar("stats_max/{}".format(i),
                            x_i.abs().max(),
                            niter)
            writer.add_scalar("stats_mean/{}".format(i),
                            x_i.mean(),
                            niter)
            writer.add_scalar("stats_var/{}".format(i),
                            x_i.var(),
                            niter)
            writer.add_scalar("stats_abs_mean/{}".format(i),
                            x_i.abs().mean(),
                            niter)
        else:
            sys.stdout.write("x_{}: max={:.2f} var={:.2f} mean={:.2f}\n".format(
                i, x_i.abs().max(), x_i.var(), x_i.mean(),
            ))
    for i, rnn in enumerate(model.rnn.rnn_lst):
        if writer is not None:
            writer.add_histogram('forget_bias/{}'.format(i),
                                rnn.bias.data[:rnn.output_size],
                                niter,
                                bins='sqrt')
            writer.add_histogram('reset_bias/{}'.format(i),
                                rnn.bias.data[rnn.output_size:],
                                niter,
                                bins='sqrt')
            writer.add_histogram('forget_weight_c/{}'.format(i),
                                rnn.weight_c.data[:rnn.output_size],
                                niter,
                                bins='sqrt')
            writer.add_histogram('reset_weight_c/{}'.format(i),
                                rnn.weight_c.data[rnn.output_size:],
                                niter,
                                bins='sqrt')

import math

import torch
import torch.optim as optim

def write_grad_norm(writer, params, step):
    for i, p in enumerate(params):
        assert p.grad is not None
        writer.add_scalar("grad_norm/{}".format(i),
            p.grad.norm(), step
        )

def write_grad_hist(writer, params, step):
    for i, p in enumerate(params):
        assert p.grad is not None
        writer.add_histogram("grad_hist/{}".format(i),
            p.grad.clone().cpu().data.numpy(),
            step, bins='sturges'
        )

def write_adam_update(writer, params, opt, step):
    assert isinstance(opt, optim.Adam)
    states = opt.state
    group = opt.param_groups[0]
    #assert not group['amsgrad']
    beta1, beta2 = group['betas']
    eps = group['eps']
    for i, p in enumerate(params):
        assert p.grad is not None
        state = states[p]
        step = state['step']
        bias_correction1 = 1 - beta1 ** step
        bias_correction2 = 1 - beta2 ** step
        exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
        denom = exp_avg_sq.sqrt().add_(eps)
        update = exp_avg/denom * (math.sqrt(bias_correction2)/bias_correction1)
        writer.add_scalar("adam_update/{}".format(i),
            update.abs().mean(), step
        )


class EffectiveRank(object):
    def __init__(self, sample_size, n, writer):
        self.writer = writer
        self.sample_size = sample_size
        self.n = n
        self.buffers = [ [] for i in range(n) ]

    def add_params(self, params, step):
        assert len(params) == self.n
        self.add_grads([p.grad for p in params], step)

    def add_grads(self, grads, step):
        assert len(grads) == self.n
        self.step = step
        for i, g in enumerate(grads):
            self.buffers[i].append(g.data.view(1, -1).clone())

        if len(self.buffers[0]) == self.sample_size:
            self.write_rank()

    def write_rank(self):
        for i, buf in enumerate(self.buffers):
            delta = torch.cat(buf)
            assert delta.dim() == 2
            t = delta.mm(delta.t())
            assert t.dim() == 2 and t.size(0) == self.sample_size
            trace = t.trace()
            ev, _ = t.eig()
            max_ev = ev.max()
            self.writer.add_scalar(
                "effective_rank/{}".format(i),
                trace/max_ev,
                self.step
            )
            print (i, trace, max_ev)
        self.buffers = [ [] for i in range(self.n) ]



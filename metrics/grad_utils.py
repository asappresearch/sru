import math

import torch
import torch.optim as optim

def write_grad_stats(writer, params, step):
    tot = 0
    i = 0
    for p in params:
        if p.grad is None: continue
        gnorm = p.grad.norm()
        tot += gnorm**2
        if p.grad.dim() != 2: continue
        i += 1
        if i>3 and i%10 != 0: continue
        writer.add_scalar("grad_norm/{}".format(i),
            gnorm, step
        )
    writer.add_scalar("grad_norm/all", tot**0.5, step)

def write_param_stats(writer, params, step):
    tot = 0
    i = 0
    for p in params:
        pnorm = p.norm()
        tot += pnorm**2
        if p.dim() != 2: continue
        i += 1
        if i>3 and i%10 != 0: continue
        writer.add_scalar("param_norm/{}".format(i),
            pnorm, step
        )
    writer.add_scalar("param_norm/all", tot**0.5, step)

def write_cstate_stats(writer, hidden, step):
    depth = hidden.size(0)
    for i in range(depth):
        #if i>2 and (i+1)%10 != 0: continue
        writer.add_scalar("state_var/{}".format(i),
            hidden[i].var(), step
        )
        writer.add_scalar("state_mean/{}".format(i),
            hidden[i].mean(), step
        )

def write_output_stats(writer, outputs, step):
    for i, output in enumerate(outputs):
        #if i>2 and (i+1)%10 != 0: continue
        writer.add_scalar("output_var/{}".format(i),
            output.var(), step
        )
        writer.add_scalar("output_mean/{}".format(i),
            output.mean(), step
        )

def write_scalar(writer, data, step, name):
    for i, val in enumerate(data):
        writer.add_scalar("{}/{}".format(name, i),
            val, step
        )

def write_hist(writer, data, step):
    for i, p in enumerate(data):
        assert p is not None
        if i>3 and (i+1)%10 != 0: continue
        writer.add_histogram("data_hist/{}".format(i),
            p.clone().cpu().numpy(),
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
        if p.grad is None: continue
        if p not in states: continue
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



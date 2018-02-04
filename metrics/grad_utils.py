
import torch

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



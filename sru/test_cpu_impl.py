import cProfile, pstats, io
import torch
from sru import SRUCell
#from sru_functional import SRUCell


def test_fwd():
    cell = SRUCell(3, 5, use_tanh=True)
    mask = torch.zeros(7, 1)
    mask[0,0]=1
    mask[6,0]=1
    x = torch.randn(7, 1, 3)
    with torch.no_grad():
        out_1 = cell(x, mask_pad=mask)
    out_2 = cell(x, mask_pad=mask)
    print (out_1)
    print ()
    print (out_2)

def test_bi_fwd():
    cell = SRUCell(5, 5, bidirectional=True)
    x = torch.randn(7, 1, 5)
    mask = torch.zeros(7, 1)
    mask[0,0]=1
    mask[6,0]=1
    with torch.no_grad():
        out_1 = cell(x)
    out_2 = cell(x)
    print (out_1)
    print ()
    print (out_2)

def profile_speed():
    bcell = SRUCell(400, 200, bidirectional=True)
    bcell.eval()
    mask = torch.zeros(200, 1)
    x = torch.randn(200, 1, 400)
    pr = cProfile.Profile()
    pr.enable()
    with torch.no_grad():
        for i in range(10):
             r = bcell(x, mask_pad=mask)
    pr.disable()
    s = io.StringIO()
    sortby = 'cumulative'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    print(s.getvalue())

    pr = cProfile.Profile()
    pr.enable()
    #with torch.no_grad():
    for i in range(10):
        r = bcell(x, mask_pad=mask)
    pr.disable()
    s = io.StringIO()
    sortby = 'cumulative'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    print(s.getvalue())

test_fwd()
test_bi_fwd()
profile_speed()

